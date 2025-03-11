import json
import pandas as pd
from datasets import load_dataset
import os 

def process_swe_bench_dataset(generated_tests_path: str, out_csv_path: str, instance_ids: list, save: bool = False):
    """
    Process and merge the SWE-Bench Verified dataset with generated tests and bad patches.
    This function will fix the `model_patch` diffs, merge the datasets, and check for missing predictions.
    """
    # Load SWE-Bench Verified dataset from Hugging Face
    swe_bench_verified = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")

    # Load Generated Tests JSONL file (treated as the predictions here)
    generated_tests = []
    with open(generated_tests_path, 'r') as f:
        for line in f:
            entry = json.loads(line.strip())
            # Fix the `model_patch` if it starts with `---`
            if entry.get('model_patch', '').startswith('---'):
                entry['model_patch'] = entry['model_patch'].replace('---', 'diff --git', 1)
                print(f"Fixed model_patch for instance_id: {entry['instance_id']}")
            generated_tests.append(entry)

    # Convert SWE-Bench Verified and Generated Tests to Pandas DataFrames
    swe_bench_df = pd.DataFrame(swe_bench_verified)
    generated_tests_df = pd.DataFrame(generated_tests)

    # Check for missing predictions by comparing `instance_id`s between the two datasets
    swe_bench_ids = set(swe_bench_df['instance_id'])
    generated_tests_ids = set(generated_tests_df['instance_id'])
    
    missing_preds = swe_bench_ids - generated_tests_ids
    if missing_preds:
        print(f"Warning: Missing predictions for {len(missing_preds)} instance IDs: {missing_preds}")

    # Rename `patch` column in SWE-Bench Verified to `gold_patch`
    swe_bench_df.rename(columns={'patch': 'gold_patch'}, inplace=True)

    # Merge SWE-Bench Verified with Generated Tests on `instance_id`
    merged_df = pd.merge(
        swe_bench_df,
        generated_tests_df[['instance_id', 'model_patch', 'model_name_or_path']],
        on='instance_id',
        how='left'
    )

    # Rename `model_patch` to `candidate_test_patch`
    merged_df.rename(columns={'model_patch': 'candidate_test_patch'}, inplace=True)

    # Load the additional CSV file with bad patches
    out_df = pd.read_csv(out_csv_path)

    # Filter rows where `bad_patch` is not empty
    out_df_filtered = out_df[out_df['bad_patch'].notna() & out_df['bad_patch'].str.strip().ne("")]

    # Merge the filtered data with the current merged DataFrame on `instance_id`
    merged_df = pd.merge(
        merged_df,
        out_df_filtered[['instance_id', 'bad_patch', 'bad_patch_author', 'Review', 'Review_Author']],
        on='instance_id',
        how='left'
    )

    # Extract `model_name_or_path` for naming the output file
    if 'model_name_or_path' in generated_tests_df.columns:
        model_name_or_path = generated_tests_df['model_name_or_path'].iloc[0]
        # Sanitize the model name for use in a file name
        sanitized_model_name = model_name_or_path.replace("/", "_").replace("\\", "_").replace(" ", "_")
    else:
        raise ValueError("`model_name_or_path` is missing from the generated tests dataset.")

    # Save the final merged dataset as CSV
    if save: 
        output_dir = "TestGeneration_Datasets/"
        output_csv_path = f"{output_dir}swe_bench_merged_{sanitized_model_name}.csv"
        os.makedirs(output_dir, exist_ok=True)

        # Save as CSV
        merged_df.to_csv(output_csv_path, index=False)

        print(f"Output saved as CSV: {output_csv_path}")
    return merged_df