./plink2 --vcf ../data/raw/Training_Data/5_Genotype_Data_All_Years.vcf --out ../data/plink/bed_geno
# https://www.cog-genomics.org/plink/2.0/filter#missing
./plink2 --pfile ../data/plink/bed_geno\
         --maf 0.05\
         --hwe 1e-50\
         --geno 0.1\
         --var-filter\
         --export A\
         --thin 0.1\
         --out ../data/plink/geno
        #  --bp-space 1000\