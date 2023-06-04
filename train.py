# python imports
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# local imports
from gsformer import GSFormer, CustomImageDataset, train_model

# constants
ROOT = os.path.join("home", "niche", "gsformer")
DIR_DATA = os.path.join(ROOT, "data", "images")
DIR_OUT = os.path.join(ROOT, "out")
PATH_MODEL = os.path.join(DIR_OUT, "gsformer.pt")

BATCH = 128
N_WORKERS = 4
DEVICE = torch.device("cuda")


def main():
    # load dataset
    loader = dict({})
    for set in ["train", "val"]:
        dataset = CustomImageDataset(os.path.join(DIR_DATA, set))
        loader[set] = DataLoader(
            dataset, batch_size=BATCH, shuffle=True, num_workers=N_WORKERS
        )

    # configure model
    model = GSFormer()
    try:
        model.load_state_dict(torch.load(PATH_MODEL))
        print("Model loaded")
    except:
        print("No model found, training new model")
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # train model
    model_ft, hist = train_model(model, loader, criterion, optimizer, DEVICE)

    # save model
    torch.save(model_ft.state_dict(), PATH_MODEL)
    # plot loss
    plt.plot(hist["train"], label="train")
    plt.plot(hist["val"], label="val")
    plt.legend()
    plt.savefig(os.path.join(DIR_OUT, "loss.png"))


if __name__ == "__main__":
    main()
