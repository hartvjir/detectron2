import os
# from urllib import request
# import tarfile

# import yaml
import numpy as np
import torch
import torch.nn as nn
# from .patch_architecture import MaterialPatchNet
from net import MobileNetV3Large
from dataset_loader import MincDataset, MincLoader, get_free_gpu
import argparse

from torch.optim.lr_scheduler import StepLR


def minc_training_loop(model: torch.nn.Module, minc_loader: MincLoader, load_model=False):
    trainloader = minc_loader.train_loader
    validloader = minc_loader.validate_loader

    if load_model:
        model.load_state_dict(torch.load("saved_model.pth"))
    if torch.cuda.is_available():
        model = model.to(device)  # cuda()

    # Declaring Criterion and Optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    scheduler = StepLR(optimizer, step_size=7, gamma=0.5)
    # Training with Validation
    epochs = 50
    min_valid_loss = np.inf

    for e in range(epochs):
        train_loss = 0.0
        ind = 1
        for data, labels in trainloader:

            # Transfer Data to GPU if available
            if torch.cuda.is_available():
                data, labels = data.to(device).requires_grad_(), labels.to(device)

            # Clear the gradients
            optimizer.zero_grad()
            # Forward Pass
            target = model(data)
            # Find the Loss
            loss = criterion(target, labels)
            # Calculate gradients
            loss.backward()
            # Update Weights
            optimizer.step()
            # Calculate Loss
            train_loss += loss.item()
            print("\r training step: "+str(ind))
            ind += 1

        valid_loss = 0.0
        model.eval()     # Optional when not using Model Specific layer
        for data, labels in validloader:
            # Transfer Data to GPU if available
            if torch.cuda.is_available():
                data, labels = data.cuda(), labels.cuda()

            # Forward Pass
            target = model(data)
            # Find the Loss
            loss = criterion(target, labels)
            # Calculate Loss
            valid_loss += loss.item()

        print(f'Epoch {e+1} \t\t Training Loss: {train_loss / len(trainloader)} \t\t Validation Loss: {valid_loss / len(validloader)}')

        if min_valid_loss > valid_loss:
            print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
            min_valid_loss = valid_loss

            # Saving State Dict
            torch.save(model.state_dict(), 'saved_model.pth')
        scheduler.step()


def main(args=None):
    model = MobileNetV3Large(n_classes=23)  # , input_size=362)
    this_dir = os.path.dirname(__file__)
    print("this_dir", this_dir)
    print("user path", os.path.expanduser('~'))
    # labels_dir = os.path.join(this_dir, "../../dataset_generation/MINC/minc-2500/labels")
    # labels_dir = os.path.join("~/Documents/ipalm-vir2020-object-category-from-image/dataset_generation/MINC/minc-2500/labels")
    labels_dir = os.path.join("home.nfs/hartvjir/MINC/minc-2500/labels")
    root_dir = os.path.join("/home.nfs/hartvjir/MINC/minc-2500/")

    minc_dataset = MincDataset(root_dir, dataset_id=2, augmentation=True)
    minc_loader = MincLoader(minc_dataset)
    model = MobileNetV3Large(n_classes=minc_dataset.num_classes)  # , input_size=362)
    minc_training_loop(model, minc_loader, args.loadmodel)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--model", nargs="?", help="")  # 0 or 1 args
    parser.add_argument("--loadmodel", type=bool, nargs="?", help="")  # 0 or 1 args
    args = parser.parse_args()
    free_id = None
    if torch.cuda.is_available():
        free_id = get_free_gpu()
        device = torch.device(free_id)  # 0 denotes id number of GPU
        print("cuda:", free_id)
        main(args)
    else:
        device = 'cpu'
        print("using cpu")

