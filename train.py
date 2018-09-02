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
import time
import argparse
import datetime
from pathlib import Path

def get_argument():
    # get argument

    parser = argparse.ArgumentParser(description="Parameter for training of network ")
    parser.add_argument("--batch_size", type=int, default=16, help="input batch size for training (default:16)")
    parser.add_argument("--epochs", type=int, default=100, help="number of epoch to train (default:100)")
    parser.add_argument("--lr", type=float, default=0.001, help="initial learning rate for training (default:0.001)")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight decay (default:0.0)")
    parser.add_argument("--dropout_ratio", type=float, default=0.0, help="dropout ratio (default:0.0)")
    parser.add_argument("--embedding_dimension", type=int, default=6, help="dimension of embedded feature (default:6)")
    parser.add_argument("--outdir_path", type=str, default="./", help="directory path of outputs")
    args = parser.parse_args()
    return args

def main(args):
    # make dataset
    trans = transforms.ToTensor()
    train_dataset = face_train_Dataset("./shiraishi_face", transform=trans)
    valid_dataset = face_train_Dataset("./shiraishi_face", transform=trans)
    train_loader = data_utils.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)
    valid_loader = data_utils.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)
    train_size = len(train_dataset)
    valid_size = len(valid_dataset)
    loaders = {"train": train_loader, "valid": valid_loader}
    dataset_sizes = {"train": train_size, "valid": valid_size}

    # make network
    c, w, h = train_dataset[0][0].size()
    net = Autoencoder()
    #net.cuda()

    # make loss function and optimizer
    #criterion = nn.BCELoss().cuda
    criterion = nn.BCELoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # initialize model state
    best_model_wts = net.state_dict()
    best_loss = 0.0

    # initialize loss
    loss_history = {"train": [], "valid":[]}

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
                inputs, _ = data

                # wrap the in valiables
                if phase == "train":
                    inputs = Variable(inputs)
                    torch.set_grad_enabled(True)
                else:
                    inputs = Variable(inputs)
                    torch.set_grad_enabled(False)

                # zero gradients
                optimizer.zero_grad()

                # forward
                _, outputs = net(inputs)
                loss = criterion(outputs, inputs)

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
    return net, loss_history

def recog(args, model_params):

    trans = transforms.ToTensor()
    valid_dataset = face_train_Dataset("./shiraishi_face", transform=trans)
    valid_loader = data_utils.DataLoader(valid_dataset, batch_size=1, shuffle=True, num_workers=1)
    valid_size = len(valid_dataset)
    loaders = {"valid": valid_loader}
    dataset_sizes = {"valid": valid_size}

    # make network
    c, w, h = valid_dataset[0][0].size()
    net = Autoencoder()
    net.load_state_dict(torch.load(model_params))

    # make loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    running_loss = 0.0
    for i, data in enumerate(loaders["valid"]):

        inputs, _ = data

        inputs = Variable(inputs)
        torch.set_grad_enabled(False)

        # zero gradients
        optimizer.zero_grad()

        # forward
        _, outputs = net(inputs)
        loss = criterion(outputs, inputs)

        running_loss += loss.item()
        visible_image = (outputs[0].numpy()*255).astype(np.uint8).transpose(1, 2, 0)
        cv2.imshow("test", visible_image)
        cv2.waitKey(300)
        epoch_loss = running_loss / dataset_sizes["valid"] * args.batch_size


if __name__ == "__main__":
    args = get_argument()
    """
    
    model_weights, loss_history = main(args)
    torch.save(model_weights.state_dict(), Path(args.outdir_path).joinpath("weight.pth"))
    training_history = np.zeros((2, args.epochs))
    for i, phase in enumerate(["train", "valid"]):
        training_history[i] = loss_history[phase]
    np.save(Path(args.outdir_path).joinpath("training_history_{}.npy".format(datetime.date.today())), training_history)

    """
    recog(args, "./weight.pth")

