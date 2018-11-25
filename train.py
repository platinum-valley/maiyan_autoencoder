import torch
import torch.nn as nn
import torch.utils.data as data_utils
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import models, datasets
from torchvision import transforms
from autoencoder import Autoencoder
from classifier import Classifier
from gan import Discriminator
from dataset import FaceDataset
from resolutor import Resolutor
from dataset import FaceResolutorDataset

import numpy as np
import os
import sys
import cv2
import pickle
import time
import argparse
import datetime
from pathlib import Path

def get_argument():
    # get argument
    parser = argparse.ArgumentParser(description="Parameter for training of network ")
    parser.add_argument("--batch_size", type=int, default=64, help="input batch size for training (default:64)")
    parser.add_argument("--epochs", type=int, default=30, help="number of epoch to train (default:20)")
    parser.add_argument("--lr", type=float, default=0.001, help="initial learning rate for training (default:0.001)")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight decay (default:0.0)")
    parser.add_argument("--dropout_ratio", type=float, default=0.0, help="dropout ratio (default:0.0)")
    parser.add_argument("--outdir_path", type=str, default="./", help="directory path of outputs")
    parser.add_argument("--gpu", action="store_true", help="using gpu")
    parser.add_argument("--task", type=str, choices=["style_transfer", "classify", "resolute"], default="",
                        help="choice training task")
    parser.add_argument("--generator_model", help="loaded generator model path")
    parser.add_argument("--discriminator_model", help="loaded discriminator model path")
    parser.add_argument("--model_type", choices=["VAE", "VAEGAN"], default="VAE", help="model architecture")
    parser.add_argument("--classifier_model", help="loaded classifier model path")
    parser.add_argument("--resolutor_model", help="loaded resolutor model path")
    parser.add_argument("--train_data", help="train dataset csv file")
    parser.add_argument("--valid_data", help="valid dataset csv file")
    parser.add_argument("--resolute_data", help="resolute dataset csv file")
    parser.add_argument("--generate_image", help="generate image path")
    args = parser.parse_args()
    return args

def train_style_transfer(args):
    if not (args.train_data and args.valid_data):
        print("must chose train_data and valid_data")
        sys.exit()

    # make dataset
    trans = transforms.ToTensor()
    train_dataset = FaceDataset(args.train_data, transform=trans)
    label_dict = train_dataset.get_label_dict()
    valid_dataset = FaceDataset(args.valid_data, transform=trans)
    valid_dataset.give_label_dict(label_dict)
    train_loader = data_utils.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    valid_loader = data_utils.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    train_size = len(train_dataset)
    valid_size = len(valid_dataset)
    loaders = {"train": train_loader, "valid": valid_loader}
    dataset_sizes = {"train": train_size, "valid": valid_size}

    if args.gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    # make network
    if args.model_type == "VAE":
        net = Autoencoder(train_dataset.label_num()).to(device)
        optimizer = optim.Adam(net.parameters(), lr=arg.lr, weight_decay=args.weight_decay)
        best_model_wts = net.state_dict()
        nest_loss =1e10
        if args.generator_model:
            net.load_state_dict(torch.load(args.generator_model))

    elif args.model_type == "VAEGAN":
        generator = Autoencoder(train_dataset.label_num()).to(device)
        discriminator = Discriminator().to(device)
        generator_optimizer = optim.Adam(generator.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        best_generator_wts = generator.state_dict()
        best_discriminator_wts = discriminator.state_dict()
        best_generator_loss = 1e10
        best_discriminator_loss = 1e10
        if args.generator_model and os.path.exists(args.generator_model):
            generator.load_state_dict(torch.load(args.generator_model))
        if args.discriminator_model and os.path.exists(args.discriminator_model):
            discriminator.load_state_dict(torch.load(args.discriminator_model))

    # make loss function and optimizer
    criterion = nn.BCELoss(reduction="sum")

    # initialize loss
    loss_history = {"train": [], "valid": []}

    # start training
    start_time = time.time()
    for epoch in range(args.epochs):
        print("epoch {}".format(epoch+1))

        for phase in ["train", "valid"]:
            if phase == "train":
                if args.model_type == "VAE":
                    net.train(True)
                elif args.model_type == "VAEGAN":
                    generator.train(True)
                    discriminator.train(True)
            else:
                if args.model_type == "VAE":
                    net.train(False)
                elif args.model_type == "VAEGAN":
                    generator.train(False)
                    discriminator.train(False)


            # initialize running loss
            running_loss = 0.0
            generator_running_loss = 0.0
            discriminator_running_loss = 0.0

            for i, data in enumerate(loaders[phase]):
                inputs, label = data

                # wrap the in valiables
                if phase == "train":
                    inputs = Variable(inputs).to(device)
                    label = Variable(label).to(device)
                    torch.set_grad_enabled(True)
                else:
                    inputs = Variable(inputs).to(device)
                    label = Variable(label).to(device)
                    torch.set_grad_enabled(False)

                # zero gradients
                if args.model_type == "VAE":
                    optimizer.zero_grad()
                    mu, var, outputs = net(inputs, label)
                    loss = loss_func(inputs, outputs, mu, var)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                    running_loss += loss.item()

                elif args.model_type == "VAEGAN":
                    real_label = Variable(torch.ones((inputs.size()[0], 1), dtype=torch.float)).to(device)
                    fake_label = Variable(torch.zeros((inputs.size()[0], 1), dtype=torch.float)).to(device)
                    discriminator_optimizer.zero_grad()

                    real_pred = discriminator(inputs)
                    real_loss = criterion(real_pred, real_label)

                    mu, var, outputs = generator(inputs, label)
                    fake_pred = discriminator(outputs.detach())
                    fake_loss = criterion(fake_pred, fake_label)

                    discriminator_loss = real_loss + fake_loss
                    if phase == "train":
                        discriminator_loss.backward()
                        discriminator_optimizer.step()

                    generator_optimizer.zero_grad()
                    generator_loss = criterion(discriminator(outputs), real_label) + loss_func(inputs, outputs, mu, var)
                    if phase == "train":
                        generator_loss.backward()
                        generator_optimizer.step()

                    discriminator_running_loss += discriminator_loss.item()
                    generator_running_loss += generator_loss.item()

            if args.model_type == "VAE":
                epoch_loss = running_loss / dataset_sizes[phase] * args.batch_size
                loss_history[phase].append(epoch_loss)

                print("{} loss {:.4f}".format(phase, epoch_loss))
                if phase == "valid" and epoch_loss < best_loss :
                    best_model_wts = net.state_dict()
                    best_loss = epoch_loss


            elif args.model_type == "VAEGAN":
                epoch_generator_loss = generator_running_loss / dataset_sizes[phase] * args.batch_size
                epoch_discriminator_loss = discriminator_running_loss /dataset_sizes[phase] * args.batch_size

                print("{} generator loss {:.4f}".format(phase, epoch_generator_loss))
                print("{} discriminator loss {:.4f}".format(phase, epoch_discriminator_loss))
                if phase == "valid" and epoch_generator_loss < best_generator_loss:
                    best_generator_wts = generator.state_dict()
                    best_generator_loss = epoch_generator_loss
                if phase == "valid" and epoch_discriminator_loss < best_discriminator_loss:
                    best_discriminator_wts = discriminator.state_dict()
                    best_generator_loss = epoch_discriminator_loss

    elapsed_time = time.time() - start_time
    print("training complete in {:.0f}s".format(elapsed_time))
    if args.model_type == "VAE":
        net.load_state_dict(best_model_wts)
        return net, loss_history, label_dict

    elif args.model_type == "VAEGAN":
        generator.load_state_dict(best_generator_wts)
        discriminator.load_state_dict(best_discriminator_wts)
        return (generator, discriminator), label_dict

def loss_func(inputs, outputs, mu, var):
    loss = nn.BCELoss(reduction="sum")
    entropy = loss(outputs, inputs)
    std = var.mul(0.5).exp_()
    eps = std.data.new(std.size()).normal_()
    z = eps.mul(std).add_(mu)
    z = torch.sum(torch.pow(z,2))
    kld = - 0.5 * torch.sum(1 + var - mu.pow(2) - var.exp())
    return entropy + kld + z

def generate(args, model_params, label_dict):
    if args.gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    if not args.generate_image:
        print("must chose generate_image")
        sys.exit()


    transform = transforms.ToTensor()
    image = transform(cv2.imread(args.generate_image))
    image = Variable(image).to(device)

    net = Autoencoder(len(label_dict)).to(device)
    net.load_state_dict(torch.load(model_params))
    torch.set_grad_enabled(False)

    mu, var = net.encode(image)
    for i in range(len(label_dict)):
        label = np.zeros(len(label_dict))
        label[i] = 1
        label = Variable(torch.tensor([label], dtype=torch.float)).to(device)
        outputs = net.generate(mu, var, label)
        label_name = [key for key, value in label_dict.items() if value == i]
        if label_name == []:
            label_name[0] = "unk"
        visible_image = (outputs[0].cpu().numpy()*255).astype(np.uint8).transpose(1, 2, 0)
        cv2.imwrite("output_img/" + label_name[0] + ".jpg", visible_image )


def recog(args, model_params, image_dir_name, label_dict):

    trans = transforms.ToTensor()
    valid_dataset = FaceDataset(image_dir_name, "./train_data.csv", transform=trans)
    valid_dataset.give_label_dict(label_dict)
    valid_loader = data_utils.DataLoader(valid_dataset, batch_size=1, shuffle=True, num_workers=1)
    valid_size = len(valid_dataset)
    loaders = {"valid": valid_loader}
    dataset_sizes = {"valid": valid_size}

    if args.gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    # make network
    c, w, h = valid_dataset[0][0].size()
    net = Autoencoder(valid_dataset.label_num() + 1).to(device)
    net.load_state_dict(torch.load(model_params))

    # make loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    running_loss = 0.0
    for i, data in enumerate(loaders["valid"]):

        inputs, label = data

        inputs = Variable(inputs).to(device)
        label = Variable(label).to(device)
        torch.set_grad_enabled(False)

        # zero gradients
        optimizer.zero_grad()

        # forward
        mu, var, outputs = net(inputs, label)
        #loss = criterion(outputs, inputs)
        loss = loss_func(inputs, outputs, mu, var)


        running_loss += loss.item()
        
        visible_image = (inputs[0].cpu().numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
        cv2.imwrite("input_img/" +  str(i) + ".jpg", visible_image)
        #cv2.imshow("test1", visible_image)
        visible_image = (outputs[0].cpu().numpy()*255).astype(np.uint8).transpose(1, 2, 0)
        cv2.imwrite("output_img/" + str(i) + ".jpg", visible_image )
        #cv2.imshow("test2", visible_image)
        #cv2.waitKey(300)
        
        epoch_loss = running_loss / dataset_sizes["valid"] * args.batch_size


def train_classifier(args):
    if not (args.train_data and args.valid_data):
        print("must chose train_data and valid_data")
        sys.exit()

    trans = transforms.ToTensor()
    train_dataset = FaceDataset(args.train_data, transform=trans)
    label_dict = train_dataset.get_label_dict()
    valid_dataset = FaceDataset(args.valid_data, transform=trans)
    valid_dataset.give_label_dict(label_dict)
    train_loader = data_utils.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)
    valid_loader = data_utils.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)
    loaders = {"train": train_loader, "valid": valid_loader}
    dataset_sizes = {"train":len(train_dataset), "valid":len(valid_dataset)}

    if args.gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    classifier = Classifier(len(label_dict)).to(device).float()
    optimizer = optim.Adam(classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_model_wts = classifier.state_dict()
    best_loss = 1e10
    if args.classifier_model and os.path.exists(args.classifier_model):
        classifier.load_state_dict(torch.load(args.classifier_model))
    criterion = nn.CrossEntropyLoss(reduction="sum")
    start_time = time.time()
    for epoch in range(args.epochs):
        print("epoch {}".format(epoch+1))

        for phase in ["train", "valid"]:
            if phase == "train":
                classifier.train(True)
            else:
                classifier.train(False)

            running_loss = 0.0
            running_acc = 0
            for i, data in enumerate(loaders[phase]):
                inputs, label = data
                inputs = Variable(inputs).to(device)
                label = Variable(label).to(device)
                if phase == "train":
                    torch.set_grad_enabled(True)
                else:
                    torch.set_grad_enabled(False)

                optimizer.zero_grad()
                pred = classifier(inputs)
                #print(pred)
                #print("pred {}".format(torch.max(pred, 1)[1]))
                #print("label {}".format(torch.max(label, 1)[1]))
                reg_loss = 0
                for param in classifier.parameters():
                    reg_loss += (param*param).sum()

                loss = criterion(pred, torch.max(label, 1)[1]) + 1e-9 * reg_loss * reg_loss
                if phase == "train":
                    loss.backward()
                    optimizer.step()
                running_loss += loss.item()
                running_acc += (torch.max(pred, 1)[1]==torch.max(label, 1)[1]).sum().item()
            epoch_loss = running_loss / dataset_sizes[phase] * args.batch_size
            epoch_acc = running_acc / dataset_sizes[phase]
            print("{} loss {:.4f}".format(phase, epoch_loss))
            print("{} acc {:.6f}".format(phase, epoch_acc))
            if phase == "valid" and epoch_loss < best_loss:
                best_model_wts = classifier.state_dict()
                best_loss = epoch_loss

    elapsed_time = time.time() - start_time
    print("training_complete in {:.0f}".format(elapsed_time))
    classifier.load_state_dict(best_model_wts)
    return classifier, label_dict

def train_resolutor(args):
    if not (args.train_data and args.valid_data):
        print("must chose train_data and valid_data")
        sys.exit()
    trans = transforms.ToTensor()
    train_dataset = FaceResolutorDataset(args.train_data, transform=trans)
    valid_dataset = FaceResolutorDataset(args.train_data, transform=trans)
    train_loader = data_utils.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)
    valid_loader = data_utils.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)
    loaders = {"train":train_loader, "valid":valid_loader}
    dataset_sizes = {"train":len(train_dataset), "valid":len(valid_dataset)}

    if args.gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    resolutor = Resolutor().to(device).float()
    optimizer = optim.Adam(resolutor.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_model_wts = resolutor.state_dict()
    best_loss = 1e10
    if args.resolutor_model and os.path.exists(args.resolutor_model):
        resolutor.load_state_dict(torch.load(args.resolutor_model))
    criterion = nn.BCELoss(reduction="sum")
    start_time = time.time()

    for epoch in range(args.epochs):
        print("epoch {}".format(epoch+1))
        for phase in ["train", "valid"]:
            if phase == "train":
                resolutor.train(True)
            else:
                resolutor.train(False)
            running_loss = 0.0
            for i, data in enumerate(loaders[phase]):
                inputs, outputs = data
                inputs = Variable(inputs).to(device)
                outputs = Variable(inputs).to(device)
                if phase == "train":
                    torch.set_grad_enabled(True)
                else:
                    torch.set_grad_enabled(False)
                optimizer.zero_grad()
                pred = resolutor(inputs)
                loss = criterion(pred, outputs)
                if phase == "train":
                    loss.backward()
                    optimizer.step()
                running_loss += loss.item()
            epoch_loss = running_loss / dataset_sizes[phase] * args.batch_size
            print("{} loss {}".format(phase, epoch_loss))
            if phase == "valid" and epoch_loss < best_loss:
                best_model_wts = resolutor.state_dict()
                best_loss = epoch_loss
    elapsed_time = time.time() - start_time
    print("training_complete in {:0f}".format(elapsed_time))
    resolutor.load_state_dict(best_model_wts)
    return resolutor

def resolute(args):
    if not args.resolute_data:
        print("must chose resolute_data")
        sys.exit()

    trans = transforms.ToTensor()
    resolute_dataset = FaceResolutorDataset(args.resolute_data, transform=trans)
    resolute_loader = data_utils.DataLoader(resolute_dataset, batch_size=1, shuffle=True, num_workers=1)

    if args.gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    if not args.resolutor_model:
        print("must chose trained resolutor model")
        sys.exit()

    resolutor = Resolutor().to(device)
    resolutor.load_state_dict(torch.load(args.resolutor_model))
    torch.set_grad_enabled(False)
    for i, data in enumerate(resolute_loader):
        inputs, outputs = data
        inputs = Variable(inputs).to(device)
        pred = resolutor(inputs)
        visible_image = (outputs[0].cpu().numpy()*255).astype(np.uint8).transpose(1,2,0)
        cv2.imwrite("output_img/{}.jpg".format(str(i).zfill(8)), visible_image)


if __name__ == "__main__":
    args = get_argument()
    with open("label.dict.pkl", "rb") as f:
        label_dict = pickle.load(f)
    generate(args, "./weight_generator2.pth", label_dict)
    """
    if args.task == "style_transfer":
        model_weights, label_dict = train_style_transfer(args)
        if args.model_type == "VAE":
            torch.save(model_weights.state_dict(), Path(args.outdir_path).joinpath("weights_aug.pth"))
        else:
            generator_weights, discriminator_weights = model_weights
            torch.save(generator_weights.state_dict(), Path(args.outdir_path).joinpath(args.generator_model))
            torch.save(discriminator_weights.state_dict(), Path(args.outdir_path).joinpath(args.discriminator_model))
        with open("label.dict.pkl", "wb")as f:
            pickle.dump(label_dict, f, pickle.HIGHEST_PROTOCOL)
    elif args.task == "classify":
        model_weights, label_dict = train_classifier(args)
        torch.save(model_weights.state_dict(), Path(args.outdir_path).joinpath(args.classifier_model))
    elif args.task == "resolute":
        model_weights = train_resolutor(args)
        torch.save(model_weights.state_dict(), Path(args.outdir_path).joinpath(args.resolutor_model))
    else :
        print("must chose task (style_transfer, classify, resolute)")
        sys.exit()

    """
    """
    model_weights, loss_history, label_dict = train_style_transfer(args)
    if args.model_type == "VAE":
        torch.save(model_weights.state_dict(), Path(args.outdir_path).joinpath("weight_aug.pth"))
    elif args.model_type == "VAEGAN":
        torch.save(model_weights[0].state_dict(), Path(args.outdir_path).joinpath("weight_generator.pth"))
        torch.save(model_weights[1].state_dict(), Path(args.outdir_path).joinpath("weight_discriminator.pth"))
    with open("label.dict.pkl", "wb") as f:
        pickle.dump(label_dict, f, pickle.HIGHEST_PROTOCOL)
    #training_history = np.zeros((2, args.epochs))
    #for i, phase in enumerate(["train", "valid"]):
    #    training_history[i] = loss_history[phase]
    #np.save(Path(args.outdir_path).joinpath("training_history_{}.npy".format(datetime.date.today())), training_history)
    
    label_dict = {}
    with open("label.dict.pkl", "rb") as f:
        label_dict = pickle.load(f)
    generate(args, "./weight_generator.pth", label_dict)
    """
    """
    model_weights, label_dict = train_classifier(args)
    torch.save(model_weights.state_dict(), Path(args.outdir_path).joinpath("weight_classifier.pth"))
    """
    """
    model_weights = train_resolutor(args)
    """
