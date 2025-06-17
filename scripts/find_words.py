import os
import re
from pathlib import Path


def find_files_containing_word(directory, word, extension=".txt"):
    matching_files = []

    # Walk through the directory
    for root, dirs, files in os.walk(directory):
        # Filter for .txt files
        for file in files:
            if file.endswith(extension):
                file_path = Path(os.path.join(root, file))
                try:
                    # Open and read the file
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        # Check if the word is in the content
                        if re.search(r"\b" + re.escape(word) + r"\b", content, re.IGNORECASE):
                            matching_files.append(file_path)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

    return matching_files


# Example usage
directory_path = "/mnt/900/input/nsfw/images/nude"  # Replace with your directory path
word_to_find = "sorry"

matching_files = find_files_containing_word(directory_path, word_to_find)

# Print the results
print(f"Files containing the word '{word_to_find}':")
for file_path in matching_files:
    print(file_path)

# Optionally, save the list to a file
with open("files_with_sorry.txt", "w") as output_file:
    for file_path in matching_files:
        output_file.write(f"{file_path}\n")
        file_path.unlink()
