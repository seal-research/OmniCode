import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Run CodeArena Benchmarks")

    # Add arguments for common parameters
    parser.add_argument("--dataset_name", required=True, help="Name of the dataset")
    parser.add_argument(
        "--predictions_path", 
        nargs="+", 
        required=True, 
        help="Paths to predictions files (.json or .jsonl). Must match the number of mode flags passed."
    )
    parser.add_argument("--max_workers", type=int, default=1, help="Number of maximum workers to use")
    parser.add_argument("--run_id", required=True, help="Run ID for the evaluation")
    parser.add_argument("--instance_ids", nargs="*", help="Optional instance IDs (supported for TestGeneration and CodeReview)")

    # Add flags for selecting the benchmark
    parser.add_argument("--TestGeneration", action="store_true", help="Run the TestGeneration benchmark")
    parser.add_argument("--CodeReview", action="store_true", help="Run the CodeReview benchmark")
    parser.add_argument("--CodeMigration", action="store_true", help="Run the CodeMigration benchmark")

    args = parser.parse_args()

    # Collect the active flags
    active_flags = []
    if args.TestGeneration:
        active_flags.append("TestGeneration")
    if args.CodeReview:
        active_flags.append("CodeReview")
    if args.CodeMigration:
        active_flags.append("CodeMigration")

    # Ensure at least one flag is provided
    if not active_flags:
        print("Error: You must specify at least one of --TestGeneration, --CodeReview, or --CodeMigration.")
        return

    # Ensure the number of predictions paths matches the number of active flags
    if len(args.predictions_path) != len(active_flags):
        print(f"Error: You provided {len(args.predictions_path)} predictions path(s), "
              f"but {len(active_flags)} mode flag(s) are active. These numbers must match.")
        return

    # Map the predictions paths to the flags
    predictions_map = dict(zip(active_flags, args.predictions_path))

    # Handle TestGeneration
    if "TestGeneration" in active_flags:
        command = (
            f"python -m run_evaluation_GenTests "
            f"--dataset_name {args.dataset_name} "
            f"--predictions_path {predictions_map['TestGeneration']} "
            f"--max_workers {args.max_workers} "
            f"--run_id {args.run_id}"
        )
        # Add instance IDs if provided
        if args.instance_ids:
            command += f" --instance_ids {' '.join(args.instance_ids)}"
        print(f"Executing TestGeneration...")
        os.system(command)

    # Handle CodeReview
    if "CodeReview" in active_flags:
        command = (
            f"python -m swebench.harness.run_evaluation "
            f"--dataset_name {args.dataset_name} "
            f"--predictions_path {predictions_map['CodeReview']} "
            f"--max_workers {args.max_workers} "
            f"--run_id {args.run_id}"
        )
        # Add instance IDs if provided
        if args.instance_ids:
            command += f" --instance_ids {' '.join(args.instance_ids)}"
        print(f"Executing CodeReview...")
        os.system(command)

    # Handle CodeMigration
    if "CodeMigration" in active_flags:
        command = (
            f"python -m code_migration_module.run "
            f"--dataset_name {args.dataset_name} "
            f"--predictions_path {predictions_map['CodeMigration']} "
            f"--max_workers {args.max_workers} "
            f"--run_id {args.run_id}"
        )
        print(f"Code Migration is not yet supported!")

if __name__ == "__main__":
    main()
