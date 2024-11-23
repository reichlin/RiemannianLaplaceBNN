#!/usr/bin/env bash
#SBATCH --mem=32GB
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=2
#SBATCH --constrain='balrog|belegost|gondor|khazadum|rivendell|shelob|shire|smaug'

source "/Midgard/home/areichlin/miniconda3/etc/profile.d/conda.sh"
conda activate hessian_torch

python3 /Midgard/home/areichlin/RiemannianLaplaceBNN/main.py --wd ${WD} --hessian_type ${HT} --prob ${P}
