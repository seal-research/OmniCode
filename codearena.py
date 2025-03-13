import argparse
from pathlib import Path
import importlib

from run_evaluation_GenTests import main as GenTestMain
from runevaluation_StyleReview import main as StyleReviewMain
import swebench
from swebench.harness.utils import str2bool
from swebench.harness.run_evaluation import main as RegularEval
from CodeArena_grading import test_passed_prefix_match, test_failed_prefix_match


CUR_DIR = Path(__file__).parent
REPO_DATA_PATH = CUR_DIR / "data/codearena_repo_data.py"
REPO_DATA = eval(REPO_DATA_PATH.read_text())

def execute_command(func, **kwargs):
    """Wrapper to execute a function safely and catch errors."""
    func(**kwargs)
    # try:
    #     func(**kwargs)
    # except Exception as e:
    #     print(f"Error while executing: {func.__name__}. Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Run CodeArena Benchmarks")

    # Add arguments for common parameters
    parser.add_argument("--dataset_name", default="data/codearena_instances.json", help="Name of the dataset")
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
        help="Optional instance IDs",
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
        "--BugFixing", action="store_true", help="Run the regular BugFixing benchmark"
    )
    parser.add_argument(
        "--TestGeneration", action="store_true", help="Run the TestGeneration benchmark"
    )
    parser.add_argument(
        "--CodeReview", action="store_true", help="Run the CodeReview benchmark"
    )
    parser.add_argument(
        "--CodeMigration", action="store_true", help="Run the CodeMigration benchmark"
    )
    parser.add_argument(
        "--StyleReview", action="store_true", help="Run the StyleReview benchmark"
    )

    # Add style review specific parameters
    parser.add_argument(
        "--min_score", type=float, default=None,
        help="Minimum acceptable pylint score (0-10) for StyleReview"
    )
    parser.add_argument(
        "--max_severity", type=str, choices=['convention', 'warning', 'error'], default=None,
        help="Maximum acceptable severity level for StyleReview"
    )

    args = parser.parse_args()

    # Collect the active flags
    active_flags = []
    if args.BugFixing:
        active_flags.append("BugFixing")
    if args.TestGeneration:
        active_flags.append("TestGeneration")
    if args.CodeReview:
        active_flags.append("CodeReview")
    if args.CodeMigration:
        active_flags.append("CodeMigration")
    if args.StyleReview:
        active_flags.append("StyleReview")

    # Ensure at least one flag is provided
    if not active_flags:
        print(
            "Error: You must specify at least one of --BugFixing, --TestGeneration, --CodeReview, --CodeMigration, or --StyleReview."
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

    # Update constants in swebench
    for instance_repo in REPO_DATA:
        swebench.versioning.constants.MAP_REPO_TO_VERSION_PATHS[instance_repo] = REPO_DATA[instance_repo]["MAP_REPO_TO_VERSION_PATHS"]
        swebench.versioning.constants.MAP_REPO_TO_VERSION_PATTERNS[instance_repo] = REPO_DATA[instance_repo]["MAP_REPO_TO_VERSION_PATTERNS"]
        swebench.harness.constants.MAP_REPO_VERSION_TO_SPECS[instance_repo] = REPO_DATA[instance_repo]["MAP_REPO_VERSION_TO_SPECS"]

        print("instance repo: ", instance_repo)
        from swebench.harness.log_parsers import parse_log_pytest, parse_log_pytest_options, parse_log_pytest_v2
        if "MAP_REPO_TO_PARSER" in REPO_DATA[instance_repo]:
            repo_log_parser = eval(REPO_DATA[instance_repo]["MAP_REPO_TO_PARSER"])
        else:
            repo_log_parser = parse_log_pytest
        
        # swebench.harness.log_parsers.MAP_REPO_TO_PARSER[instance_repo] = repo_log_parser
        swebench.harness.log_parsers.MAP_REPO_TO_PARSER["ultralytics/ultralytics"] = parse_log_pytest_v2
        swebench.harness.log_parsers.MAP_REPO_TO_PARSER["freqtrade/freqtrade"] = parse_log_pytest_v2

    
        swebench.harness.log_parsers.MAP_REPO_TO_PARSER[instance_repo] = repo_log_parser

    # monkey patch the test_passed and test_failed functions in grading.py
    swebench.harness.grading.test_passed = test_passed_prefix_match
    swebench.harness.grading.test_failed = test_failed_prefix_match

    importlib.reload(swebench)

    # Handle BugFixing
    if "BugFixing" in active_flags:
        print("Executing BugFixing...")
        execute_command(
            RegularEval,
            dataset_name=args.dataset_name,
            split="test",  # Assuming split is always 'test', can be parameterized
            instance_ids=args.instance_ids,
            predictions_path=predictions_map["BugFixing"],
            max_workers=args.max_workers,
            force_rebuild=args.force_rebuild,
            cache_level=args.cache_level,
            clean=args.clean,
            open_file_limit=args.open_file_limit,
            run_id=args.run_id,
            timeout=args.timeout
        )

    # Handle TestGeneration
    if "TestGeneration" in active_flags:
        print("Executing TestGeneration...")
        execute_command(
            GenTestMain,
            dataset_name=args.dataset_name,
            split="test",  # Assuming split is always 'test', can be parameterized
            instance_ids=args.instance_ids,
            predictions_path=predictions_map["TestGeneration"],
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
            RegularEval,
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

    # Handle StyleReview
    if "StyleReview" in active_flags:
        print("Executing StyleReview...")
        execute_command(
            StyleReviewMain,
            dataset_name=args.dataset_name,
            split="test",
            instance_ids=args.instance_ids,
            predictions_path=predictions_map["StyleReview"],
            max_workers=args.max_workers,
            force_rebuild=args.force_rebuild,
            cache_level=args.cache_level,
            clean=args.clean,
            open_file_limit=args.open_file_limit,
            run_id=args.run_id,
            timeout=args.timeout,
            min_score=args.min_score,
            max_severity=args.max_severity
        )


if __name__ == "__main__":
    main()
