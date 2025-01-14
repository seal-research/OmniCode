import argparse
import os
from run_evaluation_GenTests import main as GenTestMain
from swebench.harness.utils import str2bool
from swebench.harness.run_evaluation import main as CodeReviewMain


def execute_command(func, **kwargs):
    """Wrapper to execute a function safely and catch errors."""
    try:
        func(**kwargs)
    except Exception as e:
        print(f"Error while executing: {func.__name__}. Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Run CodeArena Benchmarks")

    # Add arguments for common parameters
    parser.add_argument("--dataset_name", required=True, help="Name of the dataset")
    parser.add_argument(
        "--predictions_path",
        nargs="+",
        required=True,
        help="Paths to predictions files (.json or .jsonl). Must match the number of mode flags passed.",
    )
    parser.add_argument(
        "--max_workers", type=int, default=1, help="Number of maximum workers to use"
    )
    parser.add_argument("--run_id", required=True, help="Run ID for the evaluation")
    parser.add_argument(
        "--instance_ids",
        nargs="*",
        help="Optional instance IDs (supported for TestGeneration and CodeReview)",
    )
    parser.add_argument(
        "--open_file_limit",
        type=int,
        default=4096,
        help="Maximum number of open files",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1_800,
        help="Timeout for individual evaluations in seconds",
    )
    parser.add_argument(
        "--force_rebuild", type=str2bool, default=False, help="Force rebuild of all images"
    )
    parser.add_argument(
        "--cache_level",
        type=str,
        choices=["none", "base", "env", "instance"],
        help="Cache level - remove images above this level",
        default="env",
    )

    parser.add_argument(
        "--clean", type=str2bool, default=False, help="Clean images above cache level"
    )

    # Add flags for selecting the benchmark
    parser.add_argument(
        "--TestGeneration", action="store_true", help="Run the TestGeneration benchmark"
    )
    parser.add_argument(
        "--CodeReview", action="store_true", help="Run the CodeReview benchmark"
    )
    parser.add_argument(
        "--CodeMigration", action="store_true", help="Run the CodeMigration benchmark"
    )

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
        print(
            "Error: You must specify at least one of --TestGeneration, --CodeReview, or --CodeMigration."
        )
        return

    # Ensure the number of predictions paths matches the number of active flags
    if len(args.predictions_path) != len(active_flags):
        print(
            f"Error: You provided {len(args.predictions_path)} predictions path(s), "
            f"but {len(active_flags)} mode flag(s) are active. These numbers must match."
        )
        return

    # Map the predictions paths to the flags
    predictions_map = dict(zip(active_flags, args.predictions_path))

    # Handle TestGeneration
    if "TestGeneration" in active_flags:
        print("Executing TestGeneration...")
        execute_command(
            GenTestMain,
            dataset_name=args.dataset_name,
        split="test",  # Assuming split is always 'test', can be parameterized
        instance_ids=args.instance_ids,
        predictions_path=args.predictions_map["TestGeneration"],
        max_workers=args.max_workers,
        force_rebuild=args.force_rebuild,
        cache_level=args.cache_level,
        clean=args.clean,
        open_file_limit=args.open_file_limit,
        run_id=args.run_id,
        timeout=args.timeout,
        )

    # Handle CodeReview
    if "CodeReview" in active_flags:
        print("Executing CodeReview...")
        execute_command(
            CodeReviewMain,
            dataset_name=args.dataset_name,
            split="test",  # Assuming split is always 'test', can be parameterized
            instance_ids=args.instance_ids,
            predictions_path=predictions_map["CodeReview"],
            max_workers=args.max_workers,
            force_rebuild=args.force_rebuild,
            cache_level=args.cache_level,
            clean=args.clean,
            open_file_limit=args.open_file_limit,
            run_id=args.run_id,
            timeout=args.timeout
        )

    # Handle CodeMigration
    if "CodeMigration" in active_flags:
        print("Executing CodeMigration...")
        print(f"Code Migration is not yet supported!")


if __name__ == "__main__":
    main()
