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
    RUN_INSTANCE_LOG_DIR,
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
from swebench.harness.docker_build import (
    build_container,
    build_env_images,
    close_logger,
    setup_logger,
)
from swebench.harness.grading import get_pred_report
from swebench.harness.test_spec import make_test_spec, TestSpec
from swebench.harness.utils import load_swebench_dataset, str2bool


class EvaluationError(Exception):
    def __init__(self, instance_id, message, logger):
        super().__init__(message)
        self.instance_id = instance_id
        self.log_file = logger.log_file
        self.logger = logger

    def __str__(self):
        log_msg = traceback.format_exc()
        self.logger.info(log_msg)
        return (
            f"{self.instance_id}: {super().__str__()}\n"
            f"Check ({self.log_file}) for more information."
        )


def run_instance(
        test_spec: TestSpec,
        pred: dict,
        rm_image: bool,
        force_rebuild: bool,
        client: docker.DockerClient,
        run_id: str,
        timeout: int = None,
    ):
    """
    Run a single instance with the given prediction.
    """
    instance_id = test_spec.instance_id
    model_name_or_path = pred.get("model_name_or_path", "None").replace("/", "__")
    log_dir = RUN_INSTANCE_LOG_DIR / run_id / model_name_or_path / instance_id
    log_dir.mkdir(parents=True, exist_ok=True)

    build_dir = INSTANCE_IMAGE_BUILD_DIR / test_spec.instance_image_key.replace(":", "__")
    image_build_link = log_dir / "image_build_dir"
    if not image_build_link.exists():
        try:
            image_build_link.symlink_to(build_dir, target_is_directory=True)
        except:
            pass
    log_file = log_dir / "run_instance.log"

    report_path = log_dir / "report.json"
    if report_path.exists():
        return instance_id, json.loads(report_path.read_text())
    logger = setup_logger(instance_id, log_file)

    container = None
    try:
        container = build_container(test_spec, client, run_id, logger, rm_image, force_rebuild)
        container.start()
        logger.info(f"Container for {instance_id} started: {container.id}")

        patch_file = Path(log_dir / "patch.diff")
        if pred["model_patch"].strip() != "":
            patch_file.write_text(pred["model_patch"] or "")
            logger.info(
                f"Intermediate patch for {instance_id} written to {patch_file}, now applying to container..."
            )
            copy_to_container(container, patch_file, Path("/tmp/patch.diff"))

            val = container.exec_run(
                "git apply --allow-empty -v /tmp/patch.diff",
                workdir=f"/testbed/{test_spec.repo.split('/')[-1]}",
                user="root",
            )
            if val.exit_code != 0:
                logger.info(f"Failed to apply patch to container, trying again...")
                val = container.exec_run(
                    "patch --batch --fuzz=5 -p1 -i /tmp/patch.diff",
                    workdir=f"/testbed/{test_spec.repo.split('/')[-1]}",
                    user="root",
                )
                if val.exit_code != 0:
                    logger.info(f"{APPLY_PATCH_FAIL}:\n{val.output.decode('utf-8')}")
                    raise EvaluationError(
                        instance_id,
                        f"{APPLY_PATCH_FAIL}:\n{val.output.decode('utf-8')}",
                        logger,
                    )
                else:
                    logger.info(f"{APPLY_PATCH_PASS}:\n{val.output.decode('utf-8')}")
            else:
                logger.info(f"{APPLY_PATCH_PASS}:\n{val.output.decode('utf-8')}")

        git_diff_output_before = (
            container.exec_run("git diff", workdir=f"/testbed/{test_spec.repo.split('/')[-1]}").output.decode("utf-8").strip()
        )
        logger.info(f"Git diff before:\n{git_diff_output_before}")

        eval_file = Path(log_dir / "eval.sh")
        eval_file.write_text(test_spec.eval_script)
        logger.info(
            f"Eval script for {instance_id} written to {patch_file}, now applying to container..."
        )
        copy_to_container(container, eval_file, Path("/eval.sh"))

        result = exec_run_with_timeout(container, "/bin/bash /eval.sh", timeout=timeout)

        git_diff_output_after = (
            container.exec_run("git diff", workdir=f"/testbed/{test_spec.repo.split('/')[-1]}").output.decode("utf-8").strip()
        )

        logger.info(f"Git diff after:\n{git_diff_output_after}")
        if git_diff_output_after != git_diff_output_before:
            logger.info(f"Git diff changed after running eval script")

        for fail_to_pass in test_spec.FAIL_TO_PASS:
            fail_to_pass = fail_to_pass.replace("/", "__")
            test_log = (
                container.exec_run(f"cat {fail_to_pass}.test.log", workdir=f"/testbed/{test_spec.repo.split('/')[-1]}").output.decode("utf-8").strip()
            )
            with open(log_dir / f"{fail_to_pass}.test.log", "w") as f:
                f.write(test_log)

        logger.info(f"Grading answer for {instance_id}...")
        report = get_pred_report(
            test_spec=test_spec,
            prediction=pred,
            log_path=log_dir,
            include_tests_status=True,
        )
        logger.info(
            f"report: {report}\n"
            f"Result for {instance_id}: resolved: {report[instance_id]['resolved']}"
        )

        with open(report_path, "w") as f:
            f.write(json.dumps(report, indent=4))
        return instance_id, report
    except EvaluationError as e:
        error_msg = (f"EvaluationError {instance_id}: {e}\n"
                     f"{traceback.format_exc()}\n"
                     f"Check ({logger.log_file}) for more information.")
        logger.info(error_msg)
        print(error_msg)
    except Exception as e:
        error_msg = (f"Error in evaluating model for {instance_id}: {e}\n"
                     f"{traceback.format_exc()}\n"
                     f"Check ({logger.log_file}) for more information.")
        logger.info(error_msg)
        print(error_msg)
    finally:
        cleanup_container(client, container, logger)
        if rm_image:
            remove_image(client, test_spec.instance_image_key, logger)
        close_logger(logger)


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
    """
    client = docker.from_env()
    test_specs = list(map(make_test_spec, instances))

    instance_image_ids = {x.instance_image_key for x in test_specs}
    existing_images = {
        tag for i in client.images.list(all=True)
        for tag in i.tags if tag in instance_image_ids
    }
    if not force_rebuild and len(existing_images):
        print(f"Found {len(existing_images)} existing instance images. Will reuse them.")

    print(f"Running {len(instances)} instances...")
    with tqdm(total=len(instances), smoothing=0) as pbar:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    run_instance,
                    test_spec,
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
            for future in as_completed(futures):
                pbar.update(1)
                try:
                    future.result()
                except Exception as e:
                    traceback.print_exc()
                    continue
    print("All instances run.")


def get_dataset_from_preds(
        dataset_name: str,
        split: str,
        instance_ids: list,
        predictions: dict,
        run_id: str,
        exclude_completed: bool = True
    ):
    """
    Return only instances that have predictions and are in the dataset.
    If instance_ids is provided, only return instances with those IDs.
    If exclude_completed is True, only return instances that have not been run yet.
    """
    dataset = load_swebench_dataset(dataset_name, split)
    dataset_ids = {i["instance_id"] for i in dataset}

    if instance_ids:
        instance_ids = set(instance_ids)
        if instance_ids - dataset_ids:
            raise ValueError(
                (
                    "Some instance IDs not found in dataset!"
                    f"\nMissing IDs:\n{' '.join(instance_ids - dataset_ids)}"
                )
            )
        dataset = [i for i in dataset if i["instance_id"] in instance_ids]

    prediction_ids = set(predictions.keys())
    if prediction_ids - dataset_ids:
        raise ValueError(
            (
                "Some prediction IDs not found in dataset!"
                f"\nMissing IDs:\n{' '.join(prediction_ids - dataset_ids)}"
            )
        )

    completed_ids = set()
    for instance in dataset:
        if instance["instance_id"] not in prediction_ids:
            continue
        prediction = predictions[instance["instance_id"]]
        report_file = (
            RUN_INSTANCE_LOG_DIR
            / run_id
            / prediction["model_name_or_path"].replace("/", "__")
            / prediction["instance_id"]
            / "report.json"
        )
        if report_file.exists():
            completed_ids.add(instance["instance_id"])

    if completed_ids and exclude_completed:
        print(f"{len(completed_ids)} instances already run, skipping...")
        dataset = [i for i in dataset if i["instance_id"] not in completed_ids]

    empty_patch_ids = {k for k, v in predictions.items() if v["model_patch"] == "" or v["model_patch"] is None}
    dataset = [i for i in dataset if i["instance_id"] in prediction_ids and i["instance_id"] not in empty_patch_ids]
    return dataset


def make_run_report(
        predictions: dict,
        full_dataset: list,
        client: docker.DockerClient,
        run_id: str
    ):
    """
    Make a final evaluation and run report of the instances that have been run.
    """
    completed_ids = set()
    resolved_ids = set()
    error_ids = set()
    unstopped_containers = set()
    unremoved_images = set()
    unresolved_ids = set()
    incomplete_ids = set()
    empty_patch_ids = set()

    for instance in full_dataset:
        instance_id = instance["instance_id"]
        if instance_id not in predictions:
            incomplete_ids.add(instance_id)
            continue
        prediction = predictions[instance_id]
        if prediction.get("model_patch", None) in ["", None]:
            empty_patch_ids.add(instance_id)
            continue
        report_file = (
            RUN_INSTANCE_LOG_DIR
            / run_id
            / prediction["model_name_or_path"].replace("/", "__")
            / prediction["instance_id"]
            / "report.json"
        )
        if report_file.exists():
            completed_ids.add(instance_id)
            report = json.loads(report_file.read_text())
            if report[instance_id]["resolved"]:
                resolved_ids.add(instance_id)
            else:
                unresolved_ids.add(instance_id)
        else:
            error_ids.add(instance_id)

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

    print(f"Total instances: {len(full_dataset)}")
    print(f"Instances submitted: {len(predictions)}")
    print(f"Instances completed: {len(completed_ids)}")
    print(f"Instances incomplete: {len(incomplete_ids)}")
    print(f"Instances resolved: {len(resolved_ids)}")
    print(f"Instances unresolved: {len(unresolved_ids)}")
    print(f"Instances with empty patches: {len(empty_patch_ids)}")
    print(f"Instances with errors: {len(error_ids)}")
    print(f"Unstopped containers: {len(unstopped_containers)}")
    print(f"Unremoved images: {len(unremoved_images)}")

    report = {
        "total_instances": len(full_dataset),
        "submitted_instances": len(predictions),
        "completed_instances": len(completed_ids),
        "resolved_instances": len(resolved_ids),
        "unresolved_instances": len(unresolved_ids),
        "empty_patch_instances": len(empty_patch_ids),
        "error_instances": len(error_ids),
        "unstopped_instances": len(unstopped_containers),
        "completed_ids": list(sorted(completed_ids)),
        "incomplete_ids": list(sorted(incomplete_ids)),
        "empty_patch_ids": list(sorted(empty_patch_ids)),
        "submitted_ids": list(sorted(predictions.keys())),
        "resolved_ids": list(sorted(resolved_ids)),
        "unresolved_ids": list(sorted(unresolved_ids)),
        "error_ids": list(sorted(error_ids)),
        "unstopped_containers": list(sorted(unstopped_containers)),
        "unremoved_images": list(sorted(unremoved_images)),
    }
    report_file = Path(
        list(predictions.values())[0]["model_name_or_path"].replace("/", "__")
        + f".{run_id}"
        + ".json"
    )
    with open(report_file, "w") as f:
        print(json.dumps(report, indent=4), file=f)
    print(f"Report written to {report_file}")


def get_gold_predictions(dataset_name: str, split: str):
    """
    Get gold predictions for the given dataset and split.
    """
    dataset = load_swebench_dataset(dataset_name, split)
    return [
        {
            "instance_id": datum["instance_id"],
            "model_patch": datum["patch"],
            "model_name_or_path": "gold",
        } for datum in dataset
    ]


def main(
        dataset_name: str,
        split: str,
        instance_ids: list,
        predictions_path: str,
        max_workers: int,
        force_rebuild: bool,
        cache_level: str,
        clean: bool,
        open_file_limit: int,
        run_id: str,
        timeout: int,
    ):
    """
    Run evaluation harness for the given dataset and predictions.
    """
    assert len(run_id) > 0, "Run ID must be provided"
    resource.setrlimit(resource.RLIMIT_NOFILE, (open_file_limit, open_file_limit))
    client = docker.from_env()

    if predictions_path == 'gold':
        print("Using gold predictions - ignoring predictions_path")
        predictions = get_gold_predictions(dataset_name, split)
    else:
        if predictions_path.endswith(".json"):
            with open(predictions_path, "r") as f:
                predictions = json.load(f)
        elif predictions_path.endswith(".jsonl"):
            with open(predictions_path, "r") as f:
                predictions = [json.loads(line) for line in f]
        else:
            raise ValueError("Predictions path must be \"gold\", .json, or .jsonl")

    # Filter predictions to include only the specified instance_ids (if provided)
    if instance_ids:
        predictions = {pred["instance_id"]: pred for pred in predictions if pred["instance_id"] in instance_ids}
    else:
        predictions = {pred["instance_id"]: pred for pred in predictions}

    dataset = get_dataset_from_preds(dataset_name, split, instance_ids, predictions, run_id)
    full_dataset = load_swebench_dataset(dataset_name, split)
    existing_images = list_images(client)
    print(f"Running {len(dataset)} unevaluated instances...")
    if not dataset:
        print("No instances to run.")
    else:
        build_env_images(client, dataset, force_rebuild, max_workers)
        run_instances(predictions, dataset, cache_level, clean, force_rebuild, max_workers, run_id, timeout)

    clean_images(client, existing_images, cache_level, clean)
    make_run_report(predictions, full_dataset, client, run_id)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_name", default="Daoguang/Multi-SWE-bench", type=str, help="Name of dataset or path to JSON file.")
    parser.add_argument("--split", type=str, default="java_verified", help="Split of the dataset")
    parser.add_argument("--instance_ids", nargs="+", type=str, help="Instance IDs to run (space separated)")
    parser.add_argument("--predictions_path", type=str, help="Path to predictions file - if 'gold', uses gold predictions", required=True)
    parser.add_argument("--max_workers", type=int, default=4, help="Maximum number of workers (should be <= 75%% of CPU cores)")
    parser.add_argument("--open_file_limit", type=int, default=4096, help="Open file limit")
    parser.add_argument("--timeout", type=int, default=1_800, help="Timeout (in seconds) for running tests for each instance")
    parser.add_argument("--force_rebuild", type=str2bool, default=False, help="Force rebuild of all images")
    parser.add_argument(
        "--cache_level",
        type=str,
        choices=["none", "base", "env", "instance"],
        help="Cache level - remove images above this level",
        default="env",
    )
    parser.add_argument("--clean", type=str2bool, default=False, help="Clean images above cache level")
    parser.add_argument("--run_id", type=str, required=True, help="Run ID - identifies the run")
    args = parser.parse_args()

    main(**vars(args))
