import json
import pandas as pd
from datasets import load_dataset

# Load SWE-Bench Verified dataset from Hugging Face
swe_bench_verified = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")

# Load Generated Tests JSONL file
generated_tests_path = "generated_tests.jsonl"
generated_tests = []
with open(generated_tests_path, 'r') as f:
    for line in f:
        generated_tests.append(json.loads(line.strip()))

# Convert SWE-Bench Verified and Generated Tests to Pandas DataFrames
swe_bench_df = pd.DataFrame(swe_bench_verified)
generated_tests_df = pd.DataFrame(generated_tests)

# Rename `patch` column in SWE-Bench Verified to `gold_patch`
swe_bench_df.rename(columns={'patch': 'gold_patch'}, inplace=True)

# Merge the dataframes on `instance_id`
merged_df = pd.merge(
    swe_bench_df,
    generated_tests_df[['instance_id', 'model_patch', 'model_name_or_path']],
    on='instance_id',
    how='left'
)

# Rename `model_patch` to `candidate_test_patch`
merged_df.rename(columns={'model_patch': 'candidate_test_patch'}, inplace=True)

# Extract `model_name_or_path` for naming the output file
if 'model_name_or_path' in generated_tests_df.columns:
    model_name_or_path = generated_tests_df['model_name_or_path'].iloc[0]
    # Sanitize the model name for use in a file name
    sanitized_model_name = model_name_or_path.replace("/", "_").replace("\\", "_").replace(" ", "_")
else:
    raise ValueError("`model_name_or_path` is missing from the generated tests dataset.")

# Save the merged dataset as CSV and JSONL
output_dir = "TestGeneration_Datasets/"
output_csv_path = f"{output_dir}swe_bench_merged_{sanitized_model_name}.csv"
output_jsonl_path = f"{output_dir}swe_bench_merged_{sanitized_model_name}.jsonl"

# Ensure the output directory exists
import os
os.makedirs(output_dir, exist_ok=True)

# Save as CSV
merged_df.to_csv(output_csv_path, index=False)

# Save as JSONL
# merged_df.to_json(output_jsonl_path, orient='records', lines=True)

print(f"Output saved as:\nCSV: {output_csv_path}")#\nJSONL: {output_jsonl_path}")
