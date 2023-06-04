import os
import numpy as np
import pandas as pd
import re
from PIL import Image

import torch
import torchvision.transforms.functional as F


# constants
ROOT = os.path.join("home", "niche", "gsformer")
DIR_DATA = os.path.join(ROOT, "data")
DIR_IMAGES = os.path.join(DIR_DATA, "images")
PATH_gparents = os.path.join(DIR_DATA, "g_parents.csv")
PATH_splits = os.path.join(DIR_DATA, "splits.csv")
PATH_ec_train = os.path.join(DIR_DATA, "6_Training_EC_Data_2014_2021.csv")
PATH_ec_test = os.path.join(DIR_DATA, "6_Testing_EC_Data_2022.csv")


def main():
    # read data
    dt_g = pd.read_csv(PATH_gparents)
    dt_split = pd.read_csv(PATH_splits)
    dt_ec = pd.concat(
        [
            pd.read_csv(PATH_ec_train),
            pd.read_csv(PATH_ec_test),
        ],
        axis=0,
    )

    # standardize each env variables
    dt_ec = standardize_dt_ec(dt_ec)
    dt_ec = dt_ec.dropna(axis=1, how="any")

    # get the column names (seq and non-seq)
    cols_seq, cols_nsq = obtain_ec_cols(dt_ec)

    # create folders
    make_folders(DIR_IMAGES)

    # iterate over each data point and generate images for genotypes, ec seq, and ec non-seq
    for i, query in dt_split.iterrows():
        print("processing %d / %d" % (i, len(dt_split)))
        y = query["Yield_Mg_ha"]
        name = str(i) + ".jpg"
        if "train" in query["split"]:
            split = "train"
        elif "test" in query["split"]:
            split = "test"
        else:
            split = "val"

        # obtain images for genotypes and ec
        try:
            img_g = query_g(query, dt_g)
            ec_img_sq, ec_img_nsq = query_ec(query, dt_ec, cols_seq, cols_nsq)
        except Exception as e:
            print(e)
            continue

        # concatenate images
        img = np.concatenate([img_g, ec_img_sq, ec_img_nsq], axis=0)
        # save image (X) and annotation (y)
        Image.fromarray(img).save(os.path.join(DIR_IMAGES, split, name))
        with open(os.path.join(DIR_IMAGES, split, "annotation.txt"), "a") as f:
            f.write("%s,%.6f\n" % (name, y))


# convert variable to images ----------------------------------------------------
def query_ec(
    query: pd.Series, dt_ec: pd.DataFrame, cols_seq: list[str], cols_nsq: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    # seq
    ec_seq = (
        dt_ec.loc[dt_ec["Env"] == query["Env"], cols_seq]
        .values.reshape((-1, 3, 3))
        .swapaxes(0, 1)
    )
    ec_img_seq = ec_seq.swapaxes(0, 1).swapaxes(1, 2)
    ec_img_seq = to_uint8(ec_img_seq)
    ec_img_seq = resize(ec_img_seq, dim=384)
    # non-seq (140 features)
    ec_img_nsq = dt_ec.loc[dt_ec["Env"] == query["Env"], cols_nsq].values.reshape(
        (10, 14)
    )
    ec_img_nsq = np.repeat(ec_img_nsq[:, :, np.newaxis], 3, axis=2)
    ec_img_nsq = resize(ec_img_nsq)
    ec_img_nsq = to_uint8(ec_img_nsq)

    # retur
    return ec_img_seq, ec_img_nsq


def query_g(query: pd.Series, dt_g: pd.DataFrame) -> np.ndarray:
    """
    Query the genotype of a given sample.

    Args:
        query: a row of the dt_split dataframe
        dt_g: the genotype of parents

    Returns:
        a 3D array of shape (3, 384, 384) representing the genotype of the sample
    """
    # get the genotype of parents
    p1 = dt_g.loc[dt_g["line"] == query["inbreds"], :].values[:, 1:]
    p2 = dt_g.loc[dt_g["line"] == query["testers"], :].values[:, 1:]
    # get the genotype of offspring
    f1 = p1 + p2
    # concatenate the genotype of parents and offspring
    g = np.concatenate([p1, p2, f1], axis=0)
    # normalize the genotype
    g = to_uint8(g)
    g = pad_g(g)
    # return the genotype
    return g


# misc ------------------------------------------------------------------
def standardize_dt_ec(dt_ec: pd.DataFrame):
    dt_ec.iloc[:, 1:] = dt_ec.iloc[:, 1:].apply(lambda x: (x - x.mean()) / x.std())
    return dt_ec


def obtain_ec_cols(dt_ec: pd.DataFrame):
    cols_ec = dt_ec.columns
    cols_seq = [col for col in cols_ec if re.findall(r".*[1-9]{1}$", col)]
    cols_seq.sort()
    cols_nsq = [col for col in cols_ec if re.findall(r".*[a-zA-Z]$", col)][1:]
    return cols_seq, cols_nsq


def make_folders(dir_images: str):
    # init folders and annotations
    if not os.path.exists(dir_images):
        os.mkdir(dir_images)
        os.mkdir(os.path.join(dir_images, "train"))
        os.mkdir(os.path.join(dir_images, "test"))
        os.mkdir(os.path.join(dir_images, "val"))
    for split in ["train", "test", "val"]:
        with open(os.path.join(dir_images, split, "annotation.txt"), "w") as f:
            f.write("")


def resize(array, dim=384):
    """
    Resize an array to a given dimension.
    """
    array_ts = torch.tensor(array).permute(2, 0, 1)
    return F.resize(array_ts, (dim, dim)).permute(1, 2, 0).numpy()


def to_uint8(array: np.ndarray) -> np.ndarray:
    """
    Convert an array to uint8.

    Args:
        array: an array of any type

    Returns:
        an array of type uint8
    """
    array = (array - array.min()) / (array.max() - array.min()) * 255
    array = array.astype(np.uint8)
    return array


def pad_g(img: np.ndarray, dim: int = 384) -> np.ndarray:
    """
    Args:
        img: a 2D array of shape (3, p)
        dim: the dimension of the output image

    Returns:
        a 3D array of shape (dim, dim, 3)
    """
    c, p = img.shape
    img = (
        np.repeat(img, (dim**2) // p + 1, axis=1)[:, : dim**2]
        .reshape(3, dim, dim)
        .swapaxes(0, 2)
        .swapaxes(0, 1)
    )
    return img


if __name__ == "__main__":
    main()
