# GSformer

## Workflow

### Step 1 - Define Data split

- `preprocessing/split.py` - get the data information: which split, env, and hybrid
  - input
    - `data/raw/Training_Data/1_Training_Trait_Data_2014_2021.csv`
    - `data/raw/Testing_Data/1_Submission_Template_2022.csv`
  - output - `data/splits.csv`

### Step 2 - Manage Genotypes

- `preprocessing/vcf2num.sh` - Convert VCF to numeric matrix
  - input - `data/raw/Training_Data/5_Genotype_Data_All_Years.vcf`
  - output - `data/plink/geno.raw`

- `preprocessing/hybrid2p.py` - Infer parental genotypes from the hybrid info provided by G2F
  - input - `data/plink/geno.raw`
  - output - `data/g_parents.csv`

- `preprocessing/synthesize_f1.py` - Synthesize back the F1 genotypes from parental genotypes
  - input - `data/g_parents.csv`
  - output - `data/g_f1.csv`

### Step 3 - Create Feature Images

- `make_images.py` - Combine genotype and EC data into feature images
  - input
    - `data/g_parents.csv`
    - `data/splits.csv`
    - `data/raw/Training_Data/6_Training_EC_Data_2014_2021.csv`
    - `data/raw/Testing_Data/6_Testing_EC_Data_2022.csv`
  - output
    - `data/images/<split>/%id.png` - feature images: 384 x 1152 x 3
    - `data/images/<split>/annotation.txt` - labels (yield)

### Step 4 - Train the GSformer

- `gsformer.py` - GSformer PyTorch module

- `train.py` - Train the model
  - input
    - `data/images/train/`
    - `data/images/val/`
  - output
    - `out/gsformer.pt` - trained model weights

### Step 5 - Make predictions

- `inference.py` - Make predictions on the test set
  - input
    - `data/images/test/`
    - `out/gsformer.pt`
  - output
    - `out/pred.csv` - raw predicted values

- `submission.py` - Format the submission file
  - input
    - `out/pred.csv`
    - `data/splits.csv`
    - `data/raw/Testing_Data/1_Submission_Template_2022.csv`
  - output
    - `out/submission.csv` - formatted submission file

## Features and labels

### Feature Images

- 26,213 SNP markers
- 567 (9 * 63) sequential EC variables (1-9 soil layers)
- 144 non-sequential EC variables

### Labels

- `data/images/<split>/annotation.txt` - 1 label (yield)

## Folder structure (non-script files)

- `data/` - data files
    - `images/` - feature images
        - `train/` - training set
        - `test/` - testing set
        - `val/` - validation set
            - `%id.png` - feature images: 384 x 1152 x 3
            - `annotation.txt` - labels (yield)

    - `plink/` - genotype data (PLINK)
        - `geno.raw` - genotype data (FID, IID, PAT, MAT, SEX, PHENOTYPE, 26213 SNPs)

    - `raw/` - original dataset provided by G2F
        - `Training_Data/` - training datasets
        - `Testing_Data/` - testing datasets

    - `g_f1.csv` - F1 genotypes (synthesized)
    - `g_parents.csv` - Parental genotypes (inferred)
    - `splits.csv` - envs and lines of each data split

- `out/` - project outputs
    - `gsformer.pt` - trained model weights
    - `pred.csv` - raw predicted values
    - `submission.csv` - formatted submission file

- `preprocessing/` - scripts for data preprocessing
