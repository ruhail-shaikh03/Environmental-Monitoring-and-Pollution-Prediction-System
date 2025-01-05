import os
import ast
import json
from glob import glob

def convert_to_valid_json(directory):
    files = glob(os.path.join(directory, "*.json"))
    for file in files:
        try:
            with open(file, "r") as f:
                # Parse the Python dictionary-like content
                content = ast.literal_eval(f.read())
                # Save as valid JSON
                with open(file, "w") as valid_json_file:
                    json.dump(content, valid_json_file, indent=4)
            print(f"Converted {file} to valid JSON.")
        except (ValueError, SyntaxError) as e:
            print(f"Error processing file {file}: {e}")

# Provide the directory where your JSON files are stored
convert_to_valid_json("data/")
