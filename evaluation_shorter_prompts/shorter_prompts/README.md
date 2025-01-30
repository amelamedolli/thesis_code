# Follow the below and execute one by one.
## Clone the repo
git clone https://github.com/amelamedolli/shorter_prompts.git

## Install the libraries
pip install  --upgrade "transformers"   "datasets"  "accelerate"  "evaluate"  "bitsandbytes"  "trl"  "peft"

pip install aiohttp numpy tqdm pytest torch

cd ../shorter_prompts


mkdir tutorial3


## If you want to do 20 completion for each sample for all the samples,
python3 automodel1.py --name "facebook/incoder-6B" --root-dataset humaneval --lang java-prompt-length-decrease1 --temperature 0.2 --batch-size 10 --completion-limit 20 --output-dir-prefix tutorial3 --input-limit 79

cd ../evaluation/src

## Evaluate
python3 main.py --dir /home/mtpgai23/evaluation_shorter_prompts/shorter_prompts/tutorial3/humaneval-java-prompt-length-decrease1-facebook_incoder_6B-0.2-reworded --output-dir /home/mtpgai23/evaluation_shorter_prompts/shorter_prompts/tutorial3/humaneval-java-prompt-length-decrease1-facebook_incoder_6B-0.2-reworded --recursive


cd ../


cd ../

## Get the score
python3 pass_k.py ./tutorial3/*

