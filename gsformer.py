# python imports
import os
import pandas as pd
import time
import copy

# torch imports
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.models import ViT_B_16_Weights, vit_b_16


class GSFormer(nn.Module):
    def __init__(self):
        super().__init__()

        self._hidden_size = 768
        self._input_size = 384
        self.block_names = ["ec_seq", "ec_nsq", "g"]
        self.n_models = len(self.block_names)
        self.vit = nn.ModuleDict()
        for block_name in self.block_names:
            self.vit[block_name] = self.make_ViT()
        self.mlp = nn.Sequential(
            nn.Linear(self._hidden_size * self.n_models, 1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 1),
        )

    def make_ViT(self):
        model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1)
        # freeze all parameters
        for param in model.parameters():
            param.requires_grad = False
        model.heads = nn.Identity()
        return model

    def check_param(self):
        print("Params to learn:")
        params_to_update = []
        for name, param in self.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t", name)

    def forward(self, x):
        x1 = self.vit["g"](x[:, :, 0 : self._input_size, :])
        x2 = self.vit["ec_seq"](x[:, :, self._input_size : 2 * self._input_size, :])
        x3 = self.vit["ec_nsq"](x[:, :, 2 * self._input_size : 3 * self._input_size, :])
        x = torch.cat((x1, x2, x3), dim=1)
        x = self.mlp(x)
        return x


class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(
            os.path.join(img_dir, "annotation.txt"), header=None
        )
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


def train_model(model, dataloaders, criterion, optimizer, device, num_epochs=2):
    # validate model
    model.float().to(device)
    model.check_param()
    since = time.time()

    val_acc_history = []
    train_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 10e10

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            # Iterate over data.
            running_loss = 0
            steps = 1
            num_batches = len(dataloaders[phase])
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.float().to(device)
                labels = labels.float().to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    loss = criterion(outputs[:, 0], labels)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # print which batch is being processed
                print(
                    "\r",
                    "Batch {steps}/{num_batches}: {loss:.4f}".format(
                        steps=steps, num_batches=num_batches, loss=loss.item()
                    ),
                    end="",
                )
                running_loss += loss.item() * inputs.size(0)
                steps += 1

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            print("")
            print("{} Loss: {:.4f}".format(phase, epoch_loss))

            # deep copy the model
            if phase == "val" and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == "val":
                val_acc_history.append(epoch_loss)
            elif phase == "train":
                train_acc_history.append(epoch_loss)

        print()

    history = dict({"train": train_acc_history, "val": val_acc_history})
    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print("Best val loss: {:4f}".format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, history


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
