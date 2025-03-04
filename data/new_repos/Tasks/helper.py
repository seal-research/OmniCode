input_file = "open-interpreter-task-instances.jsonl"   # Replace with your actual file
output_file = "local_test.jsonl" # Replace with your desired output file

with open(input_file, "r", encoding="utf-8") as infile:
    first_line = infile.readline()  # Read the first JSON object (line)

if first_line:  # Ensure the file is not empty
    with open(output_file, "a", encoding="utf-8") as outfile:
        outfile.write(first_line)  # Write the first JSON object to the new file

print("First instance copied successfully!")
# from pathlib import Path
# import json

# name = '/Users/ronitp/Documents/Cornell/Codearena/codearena/data/new_repos/Tasks/open-interpreter-task-instances.jsonl'

# dataset = json.loads(Path(name).read_text())

# print(type(dataset))

# print('test')