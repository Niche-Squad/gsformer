# python imports
import pandas as pd
import numpy as np
import os

# constants
ROOT = os.path.join("home", "niche", "gsformer")
DIR_DATA = os.path.join(ROOT, "data")
DIR_OUT = os.path.join(ROOT, "out")
PATH_TEST = os.path.join(
    DIR_DATA, "raw", "Testing_Data", "1_Submission_Template_2022.csv"
)
PATH_SPLIT = os.path.join(DIR_DATA, "splits.csv")
PATH_PRED = os.path.join(DIR_OUT, "preds.csv")
PATH_OUT = os.path.join(DIR_OUT, "submission.csv")


def main():
    # 1. load dataset
    dt_split = pd.read_csv(PATH_SPLIT).query("split == '2_test'")
    dt_pred = pd.read_csv(PATH_PRED)
    dt_test = pd.read_csv(PATH_TEST)

    # 2. calculate overall average yield
    dt_pred.loc[:, "index"] = dt_pred.iloc[:, 0].str.replace(".jpg", "").astype(int)
    avg = np.mean(dt_pred.pred.values)

    # 3. add predicted values from dt_pred to dt_split
    dt_sub = dt_split.loc[:, ["Env", "Hybrid"]]
    dt_sub.loc[:, "Yield_Mg_ha"] = avg
    for _, row in dt_pred.iterrows():
        pred = row["pred"]
        index = row["index"]
        dt_sub.loc[dt_sub.index == index, "Yield_Mg_ha"] = pred

    # 4. add predicted values to dt_test based on Env and Hybrid from dt_sub
    dt_final = pd.merge(dt_test, dt_sub, on=["Env", "Hybrid"], how="left")
    dt_final["Yield_Mg_ha"] = dt_final["Yield_Mg_ha_y"]
    dt_final = dt_final.loc[:, ["Env", "Hybrid", "Yield_Mg_ha"]]

    # 5. check for missing and fill with avg
    dt_final.loc[dt_final.Yield_Mg_ha.isna(), "Yield_Mg_ha"] = avg

    # 6. save
    dt_final.to_csv(PATH_OUT, index=False)


if __name__ == "__main__":
    main()
