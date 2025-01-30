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
    pattern = r"(?s)//.*?(?=(// >>>|$|public))"
    match = re.search(pattern, text)
    if match:
        return match.group(0).strip()
    return None


def reconstruct_text(original_text, perturbed_instruction):
    # Replace the original instruction with the perturbed one
    pattern = r"(?s)//.*?(?=(// >>>|$|public))"
    return re.sub(pattern, perturbed_instruction + '\n', original_text, count=1)


def get_adjacent_keys():
    keyboard = {
        'q': ['w', 'a'], 'w': ['q', 'e', 's'], 'e': ['w', 'r', 'd'],
        'r': ['e', 't', 'f'], 't': ['r', 'y', 'g'], 'y': ['t', 'u', 'h'],
        'u': ['y', 'i', 'j'], 'i': ['u', 'o', 'k'], 'o': ['i', 'p', 'l'],
        'p': ['o', '['], 'a': ['q', 's', 'z'], 's': ['w', 'a', 'd', 'x'],
        'd': ['e', 's', 'f', 'c'], 'f': ['r', 'd', 'g', 'v'],
        'g': ['t', 'f', 'h', 'b'], 'h': ['y', 'g', 'j', 'n'],
        'j': ['u', 'h', 'k', 'm'], 'k': ['i', 'j', 'l'],
        'l': ['o', 'k', ';'], 'z': ['a', 'x'], 'x': ['z', 's', 'c'],
        'c': ['x', 'd', 'v'], 'v': ['c', 'f', 'b'],
        'b': ['v', 'g', 'n'], 'n': ['b', 'h', 'm'],
        'm': ['n', 'j', ','], ' ': [' ']
    }
    return keyboard

def perturb_text_keyboard_typo(text, probability=0.1):
    import random
    keyboard = get_adjacent_keys()
    result = ""
    
    for char in text:
        lower_char = char.lower()
        if random.random() < probability and lower_char in keyboard:
            # Replace with adjacent key
            typo = random.choice(keyboard[lower_char])
            result += typo if char.islower() else typo.upper()
        else:
            result += char
            
    return result

# Function to modify each prompt
def modify_prompt(text):
    # Extract the instruction
    code_instruction = extract_code_instruction(text)
    
    # Perturb the instruction
    perturbed_instruction = perturb_text_keyboard_typo(code_instruction)
    
    # Reconstruct the text with the perturbed instruction
    modified_text = reconstruct_text(text, perturbed_instruction)
    
    return modified_text

# Apply the modification to the entire 'prompt' column using the map function
ds = ds.map(lambda x: {'prompt': modify_prompt(x['prompt'])}, batched=False)

username="jojo12345678910"
repo_name="humaneval-java-keyboard-typo"

ds.push_to_hub(f"{username}/{repo_name}")
