import pandas as pd
import numpy as np
import os

# constants
ROOT = os.path.join("home", "niche", "gsformer")
DIR_DATA = os.path.join(ROOT, "data")
PATH_SPLIT = os.path.join(DIR_DATA, "splits.csv")
PATH_GENO = os.path.join(DIR_DATA, "g_parents.csv")
PATH_OUT = os.path.join(DIR_DATA, "g_f1.csv")


def main():
    # split info (query)
    dt_split = pd.read_csv(PATH_SPLIT)
    dt_iter = dt_split.loc[
        :, ["Hybrid", "inbreds", "testers", "split"]
    ].drop_duplicates()
    hybrids, inbreds, testers, splits = (
        dt_iter["Hybrid"].values,
        dt_iter["inbreds"].values,
        dt_iter["testers"].values,
        dt_iter["split"].values,
    )

    # parent genotypes (reference)
    dt_p = pd.read_csv(PATH_GENO)
    ls_markers = ",".join(dt_p.columns[1:].values)
    ls_lines = dt_p["line"].values
    m = np.array(dt_p.iloc[:, 1:])

    with open(PATH_OUT, "w") as f:
        f.write("split,hybrids," + ls_markers + "\n")

        for i in range(len(dt_iter)):
            print("\rProcessing %d/%d" % (i + 1, len(dt_iter)), end="")
            hybrid, inbred, tester, split = (
                hybrids[i],
                inbreds[i],
                testers[i],
                splits[i],
            )

            if pd.isnull(tester) or pd.isnull(inbred):
                continue

            # find inbred
            i_ib = np.where(ls_lines == inbred)[0]
            i_ib = -1 if len(i_ib) == 0 else i_ib[0]
            # find tester
            i_te = np.where(ls_lines == tester)[0]
            i_te = -1 if len(i_te) == 0 else i_te[0]
            # handle missing
            if i_ib == -1 or i_te == -1:
                continue
            # assemble genotype and write
            g = m[i_ib] + m[i_te]
            g = ",".join(str(a) for a in g)
            f.write(split + "," + hybrid + "," + g + "\n")


if __name__ == "__main__":
    main()
