import json
import os
from huggingface_hub import login
from datasets import load_dataset, Dataset
import re
import random
import string

def process_json(file_path):
    try:
        # Load the JSON file
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Process each entry in the JSON
        for entry in data:
            if 'full_prompt' in entry and 'shortened_prompt' in entry:
                # Extract imports and class definition from full_prompt
                full_prompt = entry['full_prompt']
                imports_and_class = "".join(full_prompt.split('class Problem {')[0])
                public_static = "public static " + full_prompt.split('public static ')[1].split('{')[0].strip()
                
                # Update shortened_prompt
                entry['shortened_prompt'] = (
                    f"{imports_and_class}class Problem {{\n"
                    f"    {entry['shortened_prompt']}\n"
                    f"    {public_static} {{\n"
                )
        
        # Ensure the output directory exists
        output_dir = os.path.dirname(file_path)
        updated_file_path = os.path.join(output_dir, "updated_" + os.path.basename(file_path))
        
        # Write the updated JSON back to a file
        with open(updated_file_path, 'w') as f:
            json.dump(data, f, indent=4)
        
        print(f"Shortened prompts updated and saved successfully to {updated_file_path}.")
    
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
process_json("/home/mtpgai23/benchmarks/short_benchmark_humaneval.json")

# Log in to Hugging Face Hub
login("specify your huggingface token here")

# Load the updated dataset
dataset = load_dataset(
    'json',
    data_files="/home/mtpgai23/benchmarks/updated_precise_benchmarks2.json",
    split='train'
)

# To view the first few rows of the dataset
print(dataset)

# Load another dataset
ds = load_dataset("Dataset678/humaneval-java", split='test')

# Convert 'dataset' to a dictionary indexed by 'original_id'
dataset_dict = {row['original_id']: row['shortened_prompt'] for row in dataset}

# Update 'ds' prompts based on matching 'original_id' from 'dataset'
def update_prompts(row):
    original_id = row['name']  # Use 'name' from 'ds' to find corresponding 'original_id' in 'dataset'
    expanded_prompt = dataset_dict.get(original_id, None)
    if expanded_prompt:
        row['prompt'] = expanded_prompt
    return row

# Apply the update to 'ds'
ds = ds.map(update_prompts)

# Function to remove the import statement
def remove_import(example):
    example['prompt'] = example['prompt'].replace('\nimport org.javatuples.*;', '')
    return example

# Apply the transformation to remove imports
cleaned_ds = ds.map(remove_import)

# Print the cleaned prompts
print(cleaned_ds['prompt'])
print(cleaned_ds['prompt'][0])

# Push the cleaned dataset to the Hugging Face Hub
username = "Dataset678"
repo_name = "humaneval-java-prompt-length-decrease1"
cleaned_ds.push_to_hub(f"{username}/{repo_name}")
