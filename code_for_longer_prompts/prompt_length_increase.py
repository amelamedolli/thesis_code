from llama_cpp import Llama
from huggingface_hub import login

# Log in with your Hugging Face token
login("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")

from datasets import load_dataset

import re
import random
import string

ds = load_dataset("Dataset678/humaneval-java")

def extract_code_instruction(text):
    # Use regex to capture the instruction up to the first ">>>" if it exists, or the whole block otherwise
    pattern = r"(?s)//.*?(?=(public))"#r"(?s)//.*?(?=(// >>>|$|public))" #r"(?s)//.*?(?=(public))"
    match = re.search(pattern, text)
    if match:
        return match.group(0).strip()
    return None



llm = Llama.from_pretrained(
	repo_id="prithivMLmods/Llama-3.2-8B-GGUF-200K",
	filename="Llama-3.2-8B.F16.gguf", n_gpu_layers=-1,n_ctx =2048
)

def perturb_instruction(code_instruction):

  prompt = f"""
  You are specialized in enhancing and rewriting instructions for Java language. Improve the given instruction for clarity and detail without adding any extra content. Do not modify the code sample section or provide any additional code samples. Do not start with "Sure, I". Output the result only in a commented format as shown in the text. All your output should be commented and start with '//'
  // {code_instruction}
  """


  output = llm(
    f"[INST]{prompt}[/INST]", # Prompt
    max_tokens=128,  # Generate up to 512 tokens
    #stop=["</s>"],   # Example stop token - not necessarily correct for this specific model! Please check before using.
    echo=True, temperature=0.001,top_p=0.15,top_k=0      # Whether to echo the prompt
  )
  result = output['choices'][0]['text'].split('<|end_header_id|>')[-1]
  # print(result)
  return result


def reconstruct_text(original_text, perturbed_instruction):
    # Replace the original instruction with the perturbed one
    pattern = r"(?s)//.*?(?=(// >>>|$|public))"
    return re.sub(pattern, perturbed_instruction + '\n', original_text, count=1)

# Function to modify each prompt
def modify_prompt(text):
    # Extract the instruction
    
    code_instruction = extract_code_instruction(text)
    print(f"Input: {code_instruction}")
    print("=================")

    # Perturb the instruction
    perturbed_instruction = perturb_instruction(code_instruction)

    # Reconstruct the text with the perturbed instruction
    modified_text = reconstruct_text(text, perturbed_instruction)
    print(f"Output: {modified_text}")

    print("------------------")

    return modified_text

# Apply the modification to the entire 'prompt' column using the map function
ds = ds.map(lambda x: {'prompt': modify_prompt(x['prompt'])}, batched=False)
ds

username="Dataset678"
repo_name="humaneval-java-prompt-length-increase"

ds.push_to_hub(f"{username}/{repo_name}")

