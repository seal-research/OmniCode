import json

# Function to read and pretty print a singular JSON object from a JSONL file
def pretty_print_jsonl_objects(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line_number, line in enumerate(file, start=1):
                try:
                    json_object = json.loads(line.strip())  # Parse JSON object from the line
                    print(f"Object {line_number}:")
                    print(json.dumps(json_object, indent=4))  # Pretty print the JSON object
                    print("\n" + "-" * 40 + "\n")  # Separator between objects
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON on line {line_number}: {e}")
                
                break
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Example usage
if __name__ == "__main__":
    file_path = "/Users/ronitp/Documents/Cornell/Codearena/codearena/baselines/simple/baseline_generated_tests.jsonl"  # Replace with the path to your JSONL file
    pretty_print_jsonl_objects(file_path)
