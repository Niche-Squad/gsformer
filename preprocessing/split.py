# imports
import os
import pandas as pd

# constants
ROOT = os.path.join("home", "niche", "gsformer")
DIR_DATA = os.path.join(ROOT, "data")
PATH_train_y = os.path.join(
    DIR_DATA, "Training_Data", "1_Training_Trait_Data_2014_2021.csv"
)
PATH_test_y = os.path.join(DIR_DATA, "Testing_Data", "1_Submission_Template_2022.csv")
PATH_OUT = os.path.join(DIR_DATA, "splits.csv")


def main():
    # Load env and hybrids info (1_train and 2_test)
    # train
    dt_train = load_data(PATH_train_y)
    dt_train = preprocessing(dt_train, split="1_train")
    # test
    dt_test = load_data(PATH_test_y)
    dt_test = preprocessing(dt_test, split="2_test")
    # merge and preview
    data = pd.concat([dt_train, dt_test], axis=0)

    # generate validation (3_validation) and blocked (4_blocked) splits
    data.loc[data.testers == "LH195", "split"] = "3_validation"
    data.loc[data.testers == "PHT69", "split"] = "4_blocked"
    # drop uid
    data.drop(columns=["uid"])

    # match yield
    dt_yield = pd.read_csv(PATH_train_y).loc[:, ["Env", "Hybrid", "Yield_Mg_ha"]]
    # group by [Env, Hybrid] and calculate the mean: remove block effect
    dt_yield = dt_yield.dropna().groupby(["Env", "Hybrid"]).mean().reset_index()

    # merge yield to the split data frame
    data = pd.merge(data, dt_yield, how="left", on=["Env", "Hybrid"])
    data = pd.concat(
        [data.query("split != '2_test'").dropna(), data.query("split == '2_test'")],
        axis=0,
    )
    data = data.sort_values(by=["split", "Env", "Hybrid"])

    # write
    data.to_csv(PATH_OUT, index=False)


def load_data(path):
    data = pd.read_csv(path)
    data = data.loc[:, ["Env", "Hybrid"]]
    return data


def preprocessing(dt, split=None):
    # split the Hybrid column
    hybrids = dt["Hybrid"].str.split("/")
    dt["inbreds"] = hybrids.str[0]
    dt["testers"] = hybrids.str[1]
    # put NA to the missing values. e.g., "DKC67-44"
    dt["testers"] = dt["testers"].fillna("NA")
    # split the Env column
    loc_yr = dt["Env"].str.split("_")
    dt["loc"] = loc_yr.str[0]
    dt["yr"] = loc_yr.str[1]
    # cast data type
    dt["inbreds"] = dt["inbreds"].astype("string")
    dt["testers"] = dt["testers"].astype("string")
    dt["loc"] = dt["loc"].astype("string")
    dt["yr"] = dt["yr"].astype("int")
    # drop the duplicated rows
    dt.drop_duplicates(inplace=True)
    # split suffix
    if split is not None:
        dt["split"] = split
    # return
    return dt


if __name__ == "__main__":
    main()
