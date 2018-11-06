import torch
import torch.nn as nn
import torch.utils.data as data_utils
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import models, datasets
from torchvision import transforms
from autoencoder import Autoencoder
from dataset import face_train_Dataset

import numpy as np
import os
import cv2
import pickle
import time
import argparse
import datetime
from pathlib import Path

def get_argument():
    # get argument

    parser = argparse.ArgumentParser(description="Parameter for training of network ")
    parser.add_argument("--batch_size", type=int, default=64, help="input batch size for training (default:16)")
    parser.add_argument("--epochs", type=int, default=20, help="number of epoch to train (default:100)")
    parser.add_argument("--lr", type=float, default=0.001, help="initial learning rate for training (default:0.001)")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight decay (default:0.0)")
    parser.add_argument("--dropout_ratio", type=float, default=0.0, help="dropout ratio (default:0.0)")
    parser.add_argument("--embedding_dimension", type=int, default=6, help="dimension of embedded feature (default:6)")
    parser.add_argument("--outdir_path", type=str, default="./", help="directory path of outputs")
    parser.add_argument("--gpu", action="store_true", help="using gpu")
    args = parser.parse_args()
    return args

def main(args):
    # make dataset
    trans = transforms.ToTensor()
    train_dataset = face_train_Dataset("./nogizaka_face", "./data.csv", transform=trans)
    label_dict = train_dataset.get_label_dict()
    valid_dataset = face_train_Dataset("./nogi_face", "./valid_data.csv", transform=trans)
    valid_dataset.give_label_dict(label_dict)
    train_loader = data_utils.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)
    valid_loader = data_utils.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)
    train_size = len(train_dataset)
    valid_size = len(valid_dataset)
    loaders = {"train": train_loader, "valid": valid_loader}
    dataset_sizes = {"train": train_size, "valid": valid_size}

    if args.gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    print(device)
    # make network
    c, w, h = train_dataset[0][0].size()
    net = Autoencoder(train_dataset.label_num() + 1).to(device)
    #net.cuda()

    # make loss function and optimizer
    #criterion = nn.BCELoss().cuda
    criterion = nn.BCELoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # initialize model state
    best_model_wts = net.state_dict()
    best_loss = 0.0

    # initialize loss
    loss_history = {"train": [], "valid": []}

    # start training
    start_time = time.time()
    for epoch in range(args.epochs):
        print("epoch {}".format(epoch+1))

        for phase in ["train", "valid"]:
            if phase == "train":
                net.train(True)
            else:
                net.train(False)

            # initialize running loss
            running_loss = 0.0

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
                optimizer.zero_grad()

                # forward
                mu, var, outputs = net(inputs, label)
                #loss = criterion(outputs, inputs)
                loss = loss_func(inputs, outputs, mu, var)
                # backward and optimize
                if phase == "train":
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item()

            epoch_loss = running_loss / dataset_sizes[phase] * args.batch_size
            loss_history[phase].append(epoch_loss)

            print("{} loss {:.4f}".format(phase, epoch_loss))

            if phase == "valid" and epoch_loss < best_loss:
                best_model_wts = net.state_dict()

    elapsed_time = time.time() - start_time
    print("training complete in {:.0f}s".format(elapsed_time))

    net.load_state_dict(best_model_wts)
    return net, loss_history,label_dict

def loss_func(inputs, outputs, mu, var):
    loss = nn.BCELoss(reduction="sum")
    entropy = loss(outputs, inputs)
    kld = - 0.5 * torch.sum(1 + var - mu.pow(2) - var.exp())
    return entropy + kld

def recog(args, model_params, image_dir_name, label_dict):

    trans = transforms.ToTensor()
    valid_dataset = face_train_Dataset(image_dir_name, "./data.csv", transform=trans)
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


if __name__ == "__main__":
    args = get_argument()
    """
    model_weights, loss_history, label_dict = main(args)
    torch.save(model_weights.state_dict(), Path(args.outdir_path).joinpath("weight_with_celeba.pth"))
    with open("label.dict.pkl", "wb") as f:
        pickle.dump(label_dict, f, pickle.HIGHEST_PROTOCOL)
    training_history = np.zeros((2, args.epochs))
    for i, phase in enumerate(["train", "valid"]):
        training_history[i] = loss_history[phase]
    np.save(Path(args.outdir_path).joinpath("training_history_{}.npy".format(datetime.date.today())), training_history)
    """
    label_dict = {}
    with open("label.dict.pkl", "rb") as f:
        label_dict = pickle.load(f)
    recog(args, "./weight_with_celeba.pth", "nogizaka_face", label_dict)

