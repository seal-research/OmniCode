import json
from pathlib import Path

input_file = "/Users/ronitp/Documents/Cornell/Codearena/codearena/data/codearena_instances.jsonl"  # Replace with your actual file
output_file = "/Users/ronitp/Documents/Cornell/Codearena/codearena/data/codearena_instances.json"  # Replace with your desired output file
instance_index = 501  # Replace with the index of the instance you want to copy (0-based)

# with open(input_file, "r", encoding="utf-8") as infile:
#     lines = infile.readlines()
    
# if 0 <= instance_index < len(lines):  # Ensure the index is within range
#     selected_instance = json.loads(lines[instance_index])  # Parse JSON from the selected line
    
#     with open(output_file, "a", encoding="utf-8") as outfile:
#         # json.dump(selected_instance, outfile, indent=4)  # Write formatted JSON to output file
#         outfile.write(json.dumps(selected_instance) + "\n")
    
#     print(f"Instance at index {instance_index} copied successfully!")

#     output_test = [json.loads(l) for l in Path(output_file).read_text().splitlines()]

#     # output_test = json.loads(Path(output_file).read_text())
# else:
#     print("Error: Specified index is out of range.")


output_test = [json.loads(l) for l in Path(output_file).read_text().splitlines()]