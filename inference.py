# python imports
import os
import numpy as np
import torch
from torch.utils.data import DataLoader

# local imports
from gsformer import GSFormer, CustomImageDataset

# constants
ROOT = os.path.join("home", "niche", "gsformer")
DIR_DATA = os.path.join(ROOT, "data")
DIR_OUT = os.path.join(ROOT, "out")
DIR_TESTDATA = os.path.join(DIR_DATA, "images", "test")
PATH_OUT = os.path.join(DIR_OUT, "preds.csv")
PATH_MODEL = os.path.join(DIR_OUT, "gsformer.pt")

DEVICE = torch.device("cuda")
BATCH = 128
N_WORKERS = 4


def main():
    # 1. load dataset
    dataset = CustomImageDataset(DIR_TESTDATA)
    loader = DataLoader(dataset, batch_size=BATCH, shuffle=False, num_workers=N_WORKERS)

    # 2. configure model
    model = GSFormer()
    model.load_state_dict(torch.load(PATH_MODEL, map_location=DEVICE))
    model.float().to(DEVICE)
    model.eval()

    # 3. model inference
    preds = []
    with torch.no_grad():
        i = 1
        for inputs, labels in loader:
            print(f"Batch {i} of {len(loader)}")
            inputs = inputs.float().to(DEVICE)
            labels = labels.float().to(DEVICE)
            pred = model(inputs)
            preds.append(pred.cpu().numpy())
            i += 1

    # 4. save predictions
    preds = np.concatenate(preds)
    dt_out = loader.dataset.img_labels
    dt_out["pred"] = preds
    dt_out.to_csv(PATH_OUT, index=False)


if __name__ == "__main__":
    main()
