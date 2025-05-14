from __future__ import annotations

import docker
import json
import resource
import traceback

from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm

from swebench.harness.constants import (
    APPLY_PATCH_FAIL,
    APPLY_PATCH_PASS,
    INSTANCE_IMAGE_BUILD_DIR,
    KEY_INSTANCE_ID,
    RUN_EVALUATION_LOG_DIR,
)
from swebench.harness.docker_utils import (
    remove_image,
    copy_to_container,
    exec_run_with_timeout,
    cleanup_container,
    list_images,
    should_remove,
    clean_images,
)
from docker_build import (
    BuildImageError,
    build_container,
    build_env_images,
    close_logger,
    setup_logger,
)
from CodeArena_grading import get_eval_report_test_generation, get_fail_to_fail
#from swebench.swebench.harness.test_spec import make_test_spec, TestSpec
from CodeArena_test_spec import make_test_spec, TestSpec, generate_patch_lint_script
from swebench.harness.utils import str2bool
from utils import load_swebench_dataset, load_CodeArena_prediction_dataset, copy_from_container

import os
os.environ["STYLE_REVIEW"] = "1"


class EvaluationError(Exception):
    def __init__(self, instance_id, message, logger):
        super().__init__(message)
        self.super_str = super().__str__()
        self.instance_id = instance_id
        self.log_file = logger.log_file
        self.logger = logger

    def __str__(self):
        return (
            f"Evaluation error for {self.instance_id}: {self.super_str}\n"
            f"Check ({self.log_file}) for more information."
        )

def get_gold_predictions(dataset_name: str, instance_ids: list, split: str):
    """
    Get ground truth tests and their corresponding patches from the FAIL_TO_PASS section.

    Args:
        dataset_name (str): Name of the dataset
        instance_ids (list): List of instance IDs to process
        split (str): Dataset split to use

    Returns:
        list: List of dictionaries containing instance IDs, patches, and failing test information
    """
    dataset = load_swebench_dataset(dataset_name, split)
    results = []

    for datum in dataset:
        if datum[KEY_INSTANCE_ID] not in instance_ids:
            continue

        # loading gold prediction results assumes direct employment of swe-bench-verified
        result = {
            KEY_INSTANCE_ID: datum[KEY_INSTANCE_ID],
            "repo": datum["repo"],
            "base_commit": datum["base_commit"],
            "gold_patch": datum["patch"],
            "bad_patch" : 0,
            "candidate_test_patch": datum["test_patch"], # gold test patch
            "version": datum["version"],
            "model_name_or_path": "gold"
        }
        results.append(result)

    return results

def get_dataset_from_preds(
    dataset_name: str,
    split: str,
    instance_ids: list,
    run_id: str,
    exclude_completed: bool = True,
    codearena_instances: str = "data/codearena_instances.jsonl",
    generated_tests_path: str = "generated_tests.jsonl"
):
    """
    Return only instances that have predictions and are in the dataset as a list of dicts.
    If instance_ids is provided, only return instances with those IDs.
    If exclude_completed is True, only return instances that have not been run yet.
    """
    # Process the SWE-Bench dataset and merge with generated tests and bad patches
    merged_df = load_CodeArena_prediction_dataset(generated_tests_path, codearena_instances, instance_ids)

    # Now extract the merged instances from the DataFrame into the dataset
    dataset_ids = set(merged_df['instance_id'])

    if instance_ids:
        # Ensure only requested instance_ids are considered
        missing_preds = set(instance_ids) - dataset_ids
        if missing_preds:
            print(f"Warning: Missing predictions for {len(missing_preds)} instance IDs.")

        # Filter merged_df to only include requested instance_ids
        merged_df = merged_df[merged_df['instance_id'].isin(instance_ids)]

    # Check which instance IDs have already been run
    completed_ids = set()
    for _, instance in merged_df.iterrows():
        prediction = merged_df[merged_df['instance_id'] == instance['instance_id']].iloc[0]
        report_file = (
            RUN_EVALUATION_LOG_DIR
            / run_id
            / prediction["model_name_or_path"].replace("/", "__")
            / (prediction['instance_id']+"_testGeneration")
            / "report.json"
        )
        if report_file.exists():
            completed_ids.add(instance['instance_id'])

    if completed_ids and exclude_completed:
        # Filter dataset to only instances that have not been run
        print(f"{len(completed_ids)} instances already run, skipping...")
        merged_df = merged_df[~merged_df['instance_id'].isin(completed_ids)]

    # Filter dataset to only instances with predictions and non-empty patches
    merged_df = merged_df[merged_df['instance_id'].isin(dataset_ids)]

    # Convert the DataFrame to a list of dictionaries
    return merged_df.to_dict(orient="records")

def make_run_report(
        predictions: dict,
        full_dataset: list,
        client: docker.DockerClient,
        run_id: str,
        min_score: float | None = None,
        max_severity: str | None = None
    ) -> Path:
    """
    Make a final evaluation and run report of the instances that have been run based on pylint results.
    Also reports on images and containers that may still be running.

    Args:
        predictions (dict): Predictions dict generated by the model
        full_dataset (list): List of all instances
        client (docker.DockerClient): Docker client
        run_id (str): Run ID
        min_score (float, optional): Minimum acceptable pylint score (0-10)
        max_severity (str, optional): Maximum acceptable severity level ('convention', 'warning', 'error')

    Returns:
        Path to report file
    """
    # Severity levels in order of increasing severity
    severity_levels = ['convention', 'warning', 'error']

    # instantiate sets to store IDs of different outcomes
    completed_ids = set()
    successful_ids = set()
    error_ids = set()
    unstopped_containers = set()
    unremoved_images = set()
    unsuccessful_ids = set()
    incomplete_ids = set()

    # iterate through dataset and check if the instance has been run
    for instance in full_dataset:
        instance_id = instance[KEY_INSTANCE_ID]
        if instance_id not in predictions:
            incomplete_ids.add(instance_id)
            continue

        prediction = predictions[instance_id]
        report_file = (
            RUN_EVALUATION_LOG_DIR
            / run_id
            / prediction["model_name_or_path"].replace("/", "__")
            / (prediction[KEY_INSTANCE_ID]+"_styleReview")
            / "report.json"
        )

        if report_file.exists():
            completed_ids.add(instance_id)
            report = json.loads(report_file.read_text())
            instance_report = report[instance_id]

            # Evaluate success based on provided criteria
            is_successful = True

            # Check minimum score if specified
            if min_score is not None:
                score = instance_report["aggregated"].get("global_score", 0)
                if score < min_score:
                    is_successful = False

            # Check maximum severity if specified
            if max_severity is not None and is_successful:
                max_severity_index = severity_levels.index(max_severity)

                # Check if there are any issues above the maximum allowed severity
                for message in instance_report["error_messages"]:
                    for issue in message["messages"]:
                        severity = issue["type"]
                        if severity_levels.index(severity) > max_severity_index:
                            is_successful = False
                            break
                    if not is_successful:
                        break

            if is_successful:
                successful_ids.add(instance_id)
            else:
                unsuccessful_ids.add(instance_id)
        else:
            error_ids.add(instance_id)

    # get remaining images and containers
    images = list_images(client)
    test_specs = list(map(make_test_spec, full_dataset))
    for spec in test_specs:
        image_name = spec.instance_image_key
        if image_name in images:
            unremoved_images.add(image_name)
    containers = client.containers.list(all=True)
    for container in containers:
        if run_id in container.name:
            unstopped_containers.add(container.name)

    # print final report
    dataset_ids = {i[KEY_INSTANCE_ID] for i in full_dataset}
    print(f"Total instances: {len(full_dataset)}")
    print(f"Instances submitted: {len(set(predictions.keys()) & dataset_ids)}")
    print(f"Instances completed: {len(completed_ids)}")
    print(f"Instances incomplete: {len(incomplete_ids)}")
    print(f"Pylint Criteria Met: {len(successful_ids)}")
    print(f"Pylint Criteria Not Met: {len(unsuccessful_ids)}")
    print(f"Instances with errors: {len(error_ids)}")
    print(f"Unstopped containers: {len(unstopped_containers)}")
    print(f"Unremoved images: {len(unremoved_images)}")

    # write report to file
    report = {
        "total_instances": len(full_dataset),
        "submitted_instances": len(predictions),
        "completed_instances": len(completed_ids),
        "pylint_success": len(successful_ids),
        "pylint_failure": len(unsuccessful_ids),
        "error_instances": len(error_ids),
        "unstopped_instances": len(unstopped_containers),
        "completed_ids": list(sorted(completed_ids)),
        "incomplete_ids": list(sorted(incomplete_ids)),
        "submitted_ids": list(sorted(predictions.keys())),
        "successful_ids": list(sorted(successful_ids)),
        "unsuccessful_ids": list(sorted(unsuccessful_ids)),
        "error_ids": list(sorted(error_ids)),
        "unstopped_containers": list(sorted(unstopped_containers)),
        "unremoved_images": list(sorted(unremoved_images)),
        "evaluation_criteria": {
            "min_score": min_score,
            "max_severity": max_severity
        },
        "schema_version": 2,
    }

    report_file = Path(
        list(predictions.values())[0]["model_name_or_path"].replace("/", "__")
        + f".{run_id}"
        + ".json"
    )
    with open(report_file, "w") as f:
        print(json.dumps(report, indent=4), file=f)
    print(f"Report written to {report_file}")
    return report_file

def main(
        dataset_name: str = "data/codearena_instances.json",
        split: str = "test",
        instance_ids: list = None,
        predictions_path: str = None,
        max_workers: int = 4,
        force_rebuild: bool = False,
        cache_level: str = "env",
        clean: bool = False,
        open_file_limit: int = 4096,
        run_id: str = None,
        timeout: int = 1800,
        min_score: int = None,
        max_severity: int = None
    ):
    """
    Run evaluation harness for the given dataset and predictions.
    """
    # set open file limit
    assert len(run_id) > 0, "Run ID must be provided"
    resource.setrlimit(resource.RLIMIT_NOFILE, (open_file_limit, open_file_limit))
    client = docker.from_env()

    # load predictions as map of instance_id to prediction
    if predictions_path == 'gold':
        print("Using gold predictions - ignoring predictions_path")
        predictions = get_gold_predictions(dataset_name, instance_ids, split)
    else:
        if predictions_path.endswith(".json"):
            with open(predictions_path, "r") as f:
                predictions = json.load(f)
        elif predictions_path.endswith(".jsonl"):
            with open(predictions_path, "r") as f:
                predictions = [json.loads(line) for line in f]
        else:
            raise ValueError("Predictions path must be \"gold\", .json, or .jsonl")
    predictions = {pred[KEY_INSTANCE_ID]: pred for pred in predictions}

    # get dataset from predictions
    if predictions_path == 'gold':
        dataset = get_gold_predictions(dataset_name, instance_ids, split)
    else:
        dataset = get_dataset_from_preds(dataset_name, split, instance_ids, run_id=run_id, generated_tests_path=predictions_path)

    # Load full dataset
    full_dataset = load_swebench_dataset(dataset_name, split, instance_ids, full=True)

    existing_images = list_images(client)
    print(f"Running {len(dataset)} unevaluated instances...")
    if not dataset:
        print("No instances to run.")
    else:
        # build environment images + run instances
        build_env_images(client, dataset, force_rebuild, max_workers)
        run_instances(predictions, dataset, cache_level, clean, force_rebuild, max_workers, run_id, timeout)

    # clean images + make final report
    clean_images(client, existing_images, cache_level, clean)
    make_run_report(predictions, full_dataset, client, run_id, min_score=min_score, max_severity=max_severity)


def run_instance(
        test_spec: TestSpec,
        pred: dict,
        rm_image: bool,
        force_rebuild: bool,
        client: docker.DockerClient,
        run_id: str,
        timeout: int | None = None,
    ):
    """
    Run a single style evaluation instance with the given prediction.

    Args:
        test_spec (TestSpec): TestSpec instance.
        pred (dict): Prediction with model_name_or_path, model_patch, instance_id.
        rm_image (bool): Whether to remove the image after running.
        force_rebuild (bool): Whether to force rebuild the image.
        client (docker.DockerClient): Docker client.
        run_id (str): Run ID.
        timeout (int): Timeout for running tests.
    """
    # Set up logging directory
    instance_id = test_spec.instance_id
    model_name_or_path = pred.get("model_name_or_path", "None").replace("/", "__")
    log_dir = RUN_EVALUATION_LOG_DIR / run_id / model_name_or_path / (instance_id + "_styleReview")
    log_dir.mkdir(parents=True, exist_ok=True)

    # Link image build directory in the log directory
    build_dir = INSTANCE_IMAGE_BUILD_DIR / test_spec.instance_image_key.replace(":", "__")
    image_build_link = log_dir / "image_build_dir"
    if not image_build_link.exists():
        try:
            image_build_link.symlink_to(build_dir.absolute(), target_is_directory=True)
        except Exception as e:
            print(f"Error creating symlink: {e}")

    log_file = log_dir / "run_instance.log"
    report_path = log_dir / "report.json"

    # Define local files for aggregated report and error messages.
    aggregated_report_path = log_dir / "pylint_report.json"
    error_output_path = log_dir / "pylint_errors.json"

    if report_path.exists():
        return instance_id, json.loads(report_path.read_text())

    logger = setup_logger(instance_id, log_file)

    container = None
    try:
        # Build and start container.
        container = build_container(test_spec, client, run_id, logger, rm_image, force_rebuild)
        container.start()
        logger.info(f"Container for {instance_id} started: {container.id}")

        env_name = "testbed"  # Fixed environment name.
        repo_directory = f"/{env_name}"
        base_commit = test_spec.base_commit
        model_patch = pred.get("model_patch", "") or pred.get("gold_patch", "") or pred.get("candidate_test_patch", "")
        # hack for evaluating swe agent, which produces a linter-fixing patch on top of the gold patch
        second_patch = pred.get("second_patch", None)

        # Define absolute container paths.
        container_aggregated = "/tmp/pylint_aggregated.json"
        container_errors = "/tmp/pylint_errors.json"

        # Generate evaluation script with container paths.
        linting_eval_script = generate_patch_lint_script(
            repo_directory=repo_directory,
            base_commit=base_commit,
            patch=model_patch,
            pylint_output_path=container_aggregated,
            error_output_path=container_errors,
            env_name=env_name,
            second_patch=second_patch,
        )
        linting_eval_script = "\n".join(linting_eval_script)

        # Write the generated script to the log directory and copy it to the container.
        lint_eval_file = Path(log_dir / "lint_eval.sh")
        lint_eval_file.write_text(linting_eval_script)
        logger.info(f"Linting eval script written to {lint_eval_file}; copying to container...")
        copy_to_container(container, lint_eval_file, Path("/lint_eval.sh"))

        # Run the evaluation script inside the container.
        lint_output, lint_timed_out, lint_runtime = exec_run_with_timeout(
            container,
            "/bin/bash /lint_eval.sh",
            timeout
        )
        local_lint_output = log_dir / "lint_output.txt"
        with open(local_lint_output, "w") as f:
            f.write(lint_output)
            if lint_timed_out:
                f.write(f"\n\nTimeout error: {timeout} seconds exceeded.")
                raise EvaluationError(
                    instance_id,
                    f"Linting evaluation timed out after {timeout} seconds.",
                    logger,
                )
        logger.info(f"Linting runtime: {lint_runtime:_.2f} seconds")

        # Copy error messages files from the container.
        try:
            # First check if the files exist in the container
            for path in [container_errors, container_aggregated]:
                result = container.exec_run(f"test -f {path}")
                if result.exit_code != 0:
                    logger.error(f"File {path} does not exist in container")
                    continue

                # Copy the file
                if path == container_errors:
                    copy_from_container(container, Path(path), error_output_path)
                else:
                    copy_from_container(container, Path(path), aggregated_report_path)
        except Exception as e:
            logger.error(f"Copy failed: {str(e)}")

        # Parse the aggregated report
        aggregated_report = {}
        try:
            if aggregated_report_path.exists():
                aggregated_report = json.loads(aggregated_report_path.read_text())
            else:
                logger.error("Aggregated report file not found")
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in aggregated report output from container")
        except Exception as e:
            logger.error(f"Failed to read aggregated report: {str(e)}")

        # Parse error messages.
        error_messages = []
        if error_output_path.exists():
            try:
                error_messages = json.loads(error_output_path.read_text())
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON in error messages file {error_output_path}")
        else:
            logger.error("Error messages file not found")

        # Build final report.
        report = {
            instance_id: {
                "model_name": pred.get("model_name_or_path", "None"),
                "aggregated": aggregated_report,
                "error_messages": error_messages,
                "linting_runtime": lint_runtime,
                "timed_out": lint_timed_out
            }
        }
        logger.info(f"Report for {instance_id}: {report}")

        with open(report_path, "w") as f:
            f.write(json.dumps(report, indent=4))
        return instance_id, report

    except EvaluationError as e:
        error_msg = traceback.format_exc()
        logger.info(error_msg)
        print(e)
    except BuildImageError as e:
        error_msg = traceback.format_exc()
        logger.info(error_msg)
        print(e)
    except Exception as e:
        error_msg = (f"Error in evaluating model for {instance_id}: {e}\n"
                     f"{traceback.format_exc()}\n"
                     f"Check ({logger.log_file}) for more information.")
        logger.error(error_msg)
    finally:
        cleanup_container(client, container, logger)
        if rm_image:
            remove_image(client, test_spec.instance_image_key, logger)
        close_logger(logger)
    return

def run_instances(
        predictions: dict,
        instances: list,
        cache_level: str,
        clean: bool,
        force_rebuild: bool,
        max_workers: int,
        run_id: str,
        timeout: int,
    ):
    """
    Run all instances for the given predictions in parallel.

    Args:
        predictions (dict): Predictions dict generated by the model
        instances (list): List of instances
        cache_level (str): Cache level
        clean (bool): Clean images above cache level
        force_rebuild (bool): Force rebuild images
        max_workers (int): Maximum number of workers
        run_id (str): Run ID
        timeout (int): Timeout for running tests
    """
    client = docker.from_env()
    test_specs = list(map(make_test_spec, instances)) # will include inverted evaluation script for gold and bad patch


    # print number of existing instance images
    instance_image_ids = {x.instance_image_key for x in test_specs}
    existing_images = {
        tag for i in client.images.list(all=True)
        for tag in i.tags if tag in instance_image_ids
    }
    if not force_rebuild and len(existing_images):
        print(f"Found {len(existing_images)} existing instance images. Will reuse them.")

    # run instances in parallel
    print(f"Running {len(instances)} instances...")
    with tqdm(total=len(instances), smoothing=0) as pbar:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Create a future for running each instance
            futures = {
                executor.submit(
                    run_instance,
                    test_spec,
                    # TODO: Either optimize this lookup or find a more elegant way to pass corrected predictions
                    # next((item for item in instances if item["instance_id"] == test_spec.instance_id), None),
                    predictions[test_spec.instance_id],
                    should_remove(
                        test_spec.instance_image_key,
                        cache_level,
                        clean,
                        existing_images,
                    ),
                    force_rebuild,
                    client,
                    run_id,
                    timeout,
                ): None
                for test_spec in test_specs
            }
            # Wait for each future to complete
            for future in as_completed(futures):
                pbar.update(1)
                try:
                    # Update progress bar, check if instance ran successfully
                    future.result()
                except Exception as e:
                    traceback.print_exc()
                    continue
    print("All instances run.")




if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_name", default="data/codearena_instances.json", type=str, help="Name of dataset or path to JSON file.")
    parser.add_argument("--split", type=str, default="test", help="Split of the dataset")
    parser.add_argument("--instance_ids", nargs="+", type=str, help="Instance IDs to run (space separated)")
    parser.add_argument("--predictions_path", type=str, help="Path to predictions file - if 'gold', uses gold predictions", required=True)
    parser.add_argument("--max_workers", type=int, default=4, help="Maximum number of workers (should be <= 75%% of CPU cores)")
    parser.add_argument("--open_file_limit", type=int, default=4096, help="Open file limit")
    parser.add_argument(
        "--timeout", type=int, default=1_800, help="Timeout (in seconds) for running tests for each instance"
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
        "--min_score", type=float, default=None,
        help="Minimum acceptable pylint score (0-10) for StyleReview"
    )
    parser.add_argument(
        "--max_severity", type=str, choices=['convention', 'warning', 'error'], default='error',
        help="Maximum acceptable severity level for StyleReview"
    )
    parser.add_argument(
        "--clean", type=str2bool, default=False, help="Clean images above cache level"
    )
    parser.add_argument("--run_id", type=str, required=True, help="Run ID - identifies the run")
    args = parser.parse_args()

    main(**vars(args))
