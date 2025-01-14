import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Run CodeArena Benchmarks")

    # Add arguments for common parameters
    parser.add_argument("--dataset_name", required=True, help="Name of the dataset")
    parser.add_argument("--predictions_path", required=True, help="Path to predictions file (.json or .jsonl)")
    parser.add_argument("--max_workers", type=int, default=1, help="Number of maximum workers to use")
    parser.add_argument("--run_id", required=True, help="Run ID for the evaluation")
    parser.add_argument("--instance_ids", nargs="*", help="Optional instance IDs (for TestGeneration only)")

    # Add flags for selecting the benchmark
    parser.add_argument("--TestGeneration", action="store_true", help="Run the TestGeneration benchmark")
    parser.add_argument("--CodeReview", action="store_true", help="Run the CodeReview benchmark")
    parser.add_argument("--CodeMigration", action="store_true", help="Run the CodeMigration benchmark")

    args = parser.parse_args()

    # Ensure at least one flag is provided
    if not (args.TestGeneration or args.CodeReview or args.CodeMigration):
        print("Error: You must specify at least one of --TestGeneration, --CodeReview, or --CodeMigration.")
        return

    # Handle TestGeneration
    if args.TestGeneration:
        command = (
            f"python -m run_evaluation_GenTests "
            f"--dataset_name {args.dataset_name} "
            f"--predictions_path {args.predictions_path} "
            f"--max_workers {args.max_workers} "
            f"--run_id {args.run_id}"
        )
        # Add instance IDs if provided
        if args.instance_ids:
            command += f" --instance_ids {' '.join(args.instance_ids)}"
        print(f"Executing TestGeneration command:\n{command}")
        os.system(command)

    # Handle CodeReview
    if args.CodeReview:
        command = (
            f"python -m swebench.harness.run_evaluation "
            f"--dataset_name {args.dataset_name} "
            f"--predictions_path {args.predictions_path} "
            f"--max_workers {args.max_workers} "
            f"--run_id {args.run_id}"
        )
        print(f"Executing CodeReview command:\n{command}")
        os.system(command)

    # Handle CodeMigration
    if args.CodeMigration:
        print("CodeMigration is not yet supported.")

if __name__ == "__main__":
    main()
