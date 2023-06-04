# GSformer

## Workflow

### Data split

- `split.py` - get the split information: split, env, hybrids
  - input - `data/raw/Training_Data/1_Training_Trait_Data_2014_2021.csv`, `data/Testing_Data/1_Submission_Template_2022.csv`
  - output - `data/splits.csv`

- `split.r` - split the data into each split set
  - input - `data/splits.csv`, `data/g_f1.csv`, other environmental data
  - output - `data/y_%split.csv`, `data/g_%split.csv`, `data/soil_%split.csv`, `data/ec_seq_%split.csv`, `data/ec_nonseq_%split.csv`

### Genotypes

- `vcf2num.sh` - Convert VCF to numeric matrix
  - input - `data/raw/Training_Data/5_Genotype_Data_All_Years.vcf`
  - output - `data/plink/geno.raw`

- `hybrid2p.py` - Infer parental genotypes from hybrids
  - input - `data/plink/geno.raw`
  - output - `data/g_parents.csv`, `hybrid2p.out`

- `synthesize_f1.py` - Synthesize F1 genotypes from parental genotypes
  - input - `data/g_parents.csv`
  - output - `data/g_f1.csv`

## Features and labels

### Features

- `data/g_%split.csv` - 26,213 genotypes
- `data/ec_seq_%split.csv` - 567 (9 * 63) sequential EC variables (1-9 soil layers)
- `data/ec_nonseq_%split.csv` - 144 non-sequential EC variables

### Labels

- `data/y_%split.csv` - 1 label (yield)

## Other File structure

- `data/` - original data files ([one drive](https://virginiatech-my.sharepoint.com/:f:/g/personal/niche_vt_edu/EhA3OwNf6gNDhDU5ZuzQlVQBwUXu3qyLqwUc3MMj7snLRQ?e=829nLC))
  - `data/raw/Testing_Data` - testing datasets
  - `data/raw/Training_Data` - training datasets
  - `data/plink` - genotype data (PLINK)
    - `data/plink/geno.raw` - genotype data (numeric matrix)
  - `data/g_parents.csv` - Parental genotypes (synthesized)
  - `data/g_f1.csv` - F1 genotypes (synthesized)
  - `data/slits.csv` - envs and lines of each data split
  - `data/g_missing.csv` - lines with missing genotypes

- `out/` - figures (.png, .html)

- `notebooks/` - jupyter notebook (testing or visualization)
  - `notebooks/ft_corr.ipynb` - correlation between features
  - `notebooks/glimpse_G.ipynb` - explore population structure
  - `notebooks/glimpse_noG.ipynb` - check exceptions from the env variabels
  - `notebooks/split.ipynb` - split the data into training and testing sets

