# imports
import os
import pandas as pd
import numpy as np

# constants
ROOT = os.path.join("home", "niche", "gsformer")
DIR_DATA = os.path.join(ROOT, "data")
PATH_GENO = os.path.join(DIR_DATA, "plink", "geno.raw")
PATH_OUT = os.path.join(DIR_DATA, "g_parents.csv")


def main():
    dt_geno = load_geno(PATH_GENO)
    dt_long = wide2long(dt_geno)
    m_id = dt_long.columns[3:]
    lines = dt_long.line.unique()

    # write a string to file
    with open(PATH_OUT, "w") as f:
        # m_id as column names
        f.write("line," + ",".join(m_id) + "\n")
        # write each line
        for i, line in enumerate(lines):
            print("--- %s (%d/%d) ---" % (line, i, len(lines)), flush=True)
            genotypes = recursive_search(
                dt_long,
                [line],
                m=np.ones(len(m_id), dtype=int) * -1,
                u=np.array(range(len(m_id))),
            )
            f.write(line + "," + ",".join([str(i) for i in genotypes]) + "\n")


def load_geno(path):
    """
    return a dataframe with columns ["IID", "Inbreds", "Testers", "markers"]
    """
    # load data
    dt_geno = pd.read_csv(path, sep="\t")
    # split IID column to inbreds and testers
    dt_geno[["Inbreds", "Testers"]] = dt_geno.IID.str.split("/", expand=True)
    # concatenate columns and drop unnecessary columns
    return pd.concat(
        [  # iid, inbreds, testers
            dt_geno.iloc[:, [1, -2, -1]],
            # markers
            dt_geno.iloc[:, 6:-2],
        ],
        axis=1,
    )


def wide2long(data):
    """
    turn the column IID (wide, p1/p2) to line (long, p1 or p2)
    data is a dataframe with columns ["IID", "Inbreds", "Testers", "markers"]
    dt_long is a dataframe with columns ["line", "Inbreds", "Testers", "markers"]
    """
    dt_ref = pd.melt(
        data,
        id_vars=["IID"],
        value_vars=["Inbreds", "Testers"],
        var_name="type",
        value_name="line",
    )
    # generate search dataframe
    dt_long = pd.merge(dt_ref, data, on="IID", how="left")
    dt_long.drop(["IID", "type"], axis=1, inplace=True)
    dt_long.iloc[:, 3:] = dt_long.iloc[:, 3:].fillna(1)
    return dt_long


def get_genotypes(geno, g):
    """
    geno is a n x p matrix,
    n is the number of crosses containing the line,
    p is the number of markers
    g is the genotype that '1' represents
    return a list of genotypes (0/1 known, or -1 for unknown) of each marker
    """
    if g == 1:
        np_maxmin = np.array([geno.min(axis=0) == 0, geno.max(axis=0) == 2]).transpose()
    else:
        np_maxmin = np.array([geno.max(axis=0) == 2, geno.min(axis=0) == 0]).transpose()
    genotypes = [np.where(g)[0][0] if np.sum(g) else -1 for g in np_maxmin]
    return genotypes


def get_opp_lines(data_q, exc_id):
    data = data_q.loc[:, ["Inbreds", "Testers"]]
    ls_lines = np.unique(data.values.reshape(-1))
    opp_lines = [id for id in ls_lines if id not in exc_id]
    return opp_lines


def recursive_search(dt_long, target_lines, m, u, g=1, depth=0):
    """
    dt_long is a dataframe with columns ["line", "Inbreds", "Testers", "markers"]
    target_lines is a list of lines to search
    m is a vector of genotypes, -1 for unknown
    u is the positions of the unknown markers
    g is the genotype that '1' represents
    """
    dt_query = dt_long.query("line in @target_lines")
    print(
        "Depth %d (%d hybrids, %d markers)" % (depth, len(dt_query), len(u)), flush=True
    )
    gt_query = dt_query.iloc[:, 3:].values
    m[u] = get_genotypes(gt_query[:, u], g)
    u = np.where(m == -1)[0]
    if len(u) == 0:
        return m
    else:
        target_lines = get_opp_lines(dt_query, target_lines)
        return recursive_search(dt_long, target_lines, m, u, int(not g), depth + 1)


if __name__ == "__main__":
    main()
