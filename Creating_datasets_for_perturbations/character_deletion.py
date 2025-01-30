from huggingface_hub import login

# Log in with your Hugging Face token
login("specify your huggingface token here")

from datasets import load_dataset
import re
import random
import string

ds = load_dataset("Dataset678/humaneval-java")

def extract_code_instruction(text):
    # Use regex to capture the instruction up to the first ">>>" if it exists, or the whole block otherwise
    pattern = r"(?s)//.*?(?=(// >>>|$|public))"
    match = re.search(pattern, text)
    if match:
        return match.group(0).strip()
    return None


def reconstruct_text(original_text, perturbed_instruction):
    # Replace the original instruction with the perturbed one
    pattern = r"(?s)//.*?(?=(// >>>|$|public))"
    return re.sub(pattern, perturbed_instruction + '\n', original_text, count=1)



def perturb_text_char_deletion(text, probability=0.1):
    import random
    result = ""
    
    for char in text:
        if char.isalpha():  # Only process alphabets
            if random.random() >= probability:
                result += char
        else:
            result += char  # Keep non-alphabets unchanged
            
    return result if result else text 

# Function to modify each prompt
def modify_prompt(text):
    # Extract the instruction
    code_instruction = extract_code_instruction(text)
    
    # Perturb the instruction
    perturbed_instruction = perturb_text_char_deletion(code_instruction)
    
    # Reconstruct the text with the perturbed instruction
    modified_text = reconstruct_text(text, perturbed_instruction)
    
    return modified_text

# Apply the modification to the entire 'prompt' column using the map function
ds = ds.map(lambda x: {'prompt': modify_prompt(x['prompt'])}, batched=False)

ds
username="jojo12345678910"
repo_name="humaneval-java-char-deletion2"

ds.push_to_hub(f"{username}/{repo_name}")
