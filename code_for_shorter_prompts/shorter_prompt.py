import os
import sys
import pandas as pd
import json
from tqdm import tqdm


def initialize_json_file(json_output_path):
    """
    Initialize a JSON file to store results with context-aware constraints.
    If the file already exists, it reads the existing content to append new entries later.
    If it doesn't exist, it creates a new file and adds the opening array bracket '['.
    """
    if not os.path.exists(json_output_path):
       with open(json_output_path, "w") as json_file:
        json_file.write("[\n")
       print(f"Initialized JSON file for output: {json_output_path}")
    else:
         print(f"JSON file already exists, will append to: {json_output_path}")


def append_to_json_file(json_output_path, entry):
    """
    Append a single entry to the JSON file as a new object in the array.
    """
    with open(json_output_path, "a") as json_file:
        json.dump(entry, json_file, indent=4)
        json_file.write(",\n")  # Add a comma and newline to separate entries
    print(f"Appended entry to JSON file for original_id: {entry['original_id']}")


def finalize_json_file(json_output_path):
    """
    Properly close the JSON array in the file by removing the last comma and adding the closing bracket ']'.
    This function only works if the JSON file is being finalized (i.e., no more appending).
    """
    with open(json_output_path, "rb+") as json_file:
        json_file.seek(-2, os.SEEK_END)  # Move to the last comma
        json_file.truncate()  # Remove the last comma
        json_file.write(b"\n]")  # Close the JSON array
    print(f"Finalized JSON file: {json_output_path}")



csv_file = "/home/mtpgai23/benchmarks/humaneval.csv"  # CSV file containing the full_prompt
json_output_path = "/home/mtpgai23/benchmarks/short_benchmark_humaneval.json"

def simplify_prompt(full_prompt: str) -> str:
    """
    Simplifies a full coding prompt by extracting the core task description and any relevant example cases.
    This includes extracting the core task from comments, handling various comment formats.
    """
    # Split the prompt into lines
    lines = full_prompt.split("\n")
    
    # Initialize variables for key parts of the prompt
    core_task = []
    examples = []
    task_found = False
    inside_comment_block = False
    task_keywords = ["how many", "return", "calculate", "find", "determine", "compute", "task", "solve"]

    # Iterate through the lines to extract important information
    for line in lines:
        line = line.strip()

        # Detect multi-line comment blocks (usually contain the task description)
        if line.startswith("/*") or line.startswith("/"):
            inside_comment_block = True
            continue

        if inside_comment_block and line.endswith("*/"):
            inside_comment_block = False
            continue

        # If inside a comment block or a comment line
        if inside_comment_block or line.startswith("//") or line.startswith("*"):
            # Check for task-related keywords
            if any(keyword in line.lower() for keyword in task_keywords):
                core_task.append(line.strip("* ").strip())
                task_found = True

        # Collect example lines (e.g., test cases or example inputs/outputs)
        if "example" in line.lower() or ">>> " in line.lower() or "returns" in line.lower() or "n0" in line.lower():
            examples.append(line.strip("* ").strip())

    # If no task-related information was found but there is a description, extract the description
    if not core_task and not task_found:
        for line in lines:
            if line.startswith("public"):
                # Extract method signature or description of the method
                core_task.append("Method definition or task description present, but details are unclear.")

    # Combine the core task and examples into the shortened prompt
    simplified_prompt = "\n".join(core_task)
    if examples:
        simplified_prompt += "\n" + "\n".join(examples)
    
    return simplified_prompt.strip()


def shortened_prompts(csv_file, json_output_path):
    """
    Process each row of the CSV file, simplify the full_prompt content, and save the results to a JSON file.
    """
    print(f"Reading CSV file: {csv_file}")
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} rows from CSV.")

    # Iterate through each row of the CSV file
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        benchmark_name = row['name']
        method_name = row['method_name']
        original_id = row['original_id']
        full_prompt = row['full_prompt']

        print(f"Processing row {index+1}/{len(df)}: Benchmark: {benchmark_name}, Method: {method_name}, Original ID: {original_id}")

        # Create a more concise version of the prompt
        shortened_prompt = simplify_prompt(full_prompt)

        # Create a single JSON entry with the original and shortened prompt
        entry = {
            "name": benchmark_name,
            "method_name": method_name,
            "original_id": original_id,
            "full_prompt": full_prompt,
            "shortened_prompt": shortened_prompt  # Save the original prompt and the shortened one
        }

        # Append the entry to the JSON file
        append_to_json_file(json_output_path, entry)

        print(f"Successfully processed row {index+1}: Original ID: {original_id}")

    print(f"Finished processing {len(df)} rows from {csv_file}.")

# Manually specify the benchmark names to filter CSV files
benchmark_name = input("humaneval").strip()


# Initialize JSON file
initialize_json_file(json_output_path)


# Process CSV and apply context-aware constraints using LLM
shortened_prompts(csv_file, json_output_path)

# Finalize JSON file
finalize_json_file(json_output_path)

print("Process completed successfully!")