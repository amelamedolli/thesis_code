# Follow the below and execute one by one.
## Clone the repo
git clone https://github.com/amelamedolli/perturbation.git

## Install the libraries
pip install  --upgrade "transformers"   "datasets"  "accelerate"  "evaluate"  "bitsandbytes"  "trl"  "peft"

pip install aiohttp numpy tqdm pytest torch

cd ../code_perturbation


1. mkdir tutorial
2. mkdir tutorial1
3. mkdir tutorial2



## If you want to do 20 completion for each sample for all the sample

1. For Character Deletion Perturbation:
   
python3 automodel1.py --name "facebook/incoder-6B" --root-dataset humaneval --lang java-char-deletion --temperature 0.2 --batch-size 10 --completion-limit 20 --output-dir-prefix tutorial --input-limit 79

2. For Keyboard-Typo Perturbation:
   
python3 automodel1.py --name "facebook/incoder-6B" --root-dataset humaneval --lang java-keyboard-typo --temperature 0.2 --batch-size 10 --completion-limit 20 --output-dir-prefix tutorial1 --input-limit 79

3. For Space Deletion Perturbation:

python3 automodel1.py --name "facebook/incoder-6B" --root-dataset humaneval --lang java-space-deletion --temperature 0.2 --batch-size 10 --completion-limit 20 --output-dir-prefix tutorial2 --input-limit 79

cd ../evaluation/src

## Evaluate
1. python3 main.py --dir /home/mtpgai23/evaluation_perturbations/perturbation/tutorial/humaneval-java-char-deletion-facebook_incoder_6B-0.2-reworded --output-dir /home/mtpgai23/evaluation_perturbations/perturbation/tutorial/humaneval-java-char-deletion-facebook_incoder_6B-0.2-reworded --recursive
   
2. python3 main.py --dir /home/mtpgai23/evaluation_perturbations/perturbation/tutorial1/humaneval-java-keyboard-typo-facebook_incoder_6B-0.2-reworded --output-dir /home/mtpgai23/evaluation_perturbations/perturbation/tutorial1/humaneval-java-keyboard-typo-facebook_incoder_6B-0.2-reworded --recursive

3. python3 main.py --dir /home/mtpgai23/evaluation_perturbations/perturbation/tutorial2/humaneval-java-space-deletion-facebook_incoder_6B-0.2-reworded --output-dir /home/mtpgai23/evaluation_perturbations/perturbation/tutorial2/humaneval-java-space-deletion-facebook_incoder_6B-0.2-reworded --recursive


cd ../


cd ../

## Get the score
1. python3 pass_k.py ./tutorial/*
2. python3 pass_k.py ./tutorial1/*
3. python3 pass_k.py ./tutorial2/*

