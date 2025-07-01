from pathlib import Path
from typing import Any
from enum import Enum

from swebench.harness.constants import (
    APPLY_PATCH_FAIL,
    APPLY_PATCH_PASS,
    FAIL_TO_FAIL,
    FAIL_TO_PASS,
    KEY_INSTANCE_ID,
    PASS_TO_FAIL,
    PASS_TO_PASS,
    RESET_FAILED,
    TESTS_ERROR,
    TESTS_TIMEOUT,
    ResolvedStatus,
    TestStatus,
)
from utils import merge_and_unpack
# from swebench.harness.test_spec import TestSpec
from swebench.harness.test_spec import test_spec as TestSpec
from swebench.harness.log_parsers import MAP_REPO_TO_PARSER

class TestedStatus(Enum):
    FAIL = "GOLD_FAIL_BAD_FAIL"
    SUCCESS_GOLD = "GOLD_SUCCESS_BAD_FAIL"
    SUCCESS_BAD = "GOLD_FAIL_BAD_SUCCESS"
    SUCCESS = "GOLD_SUCCESS_BAD_SUCCESS"


def test_passed_prefix_match(case: str, sm: dict[str, str]) -> bool:
    if case in sm:
        return sm[case] in [TestStatus.PASSED.value, TestStatus.XFAIL.value]

    # alternate version which can handle case where "case" is something like "test/test_InfoExtractor"
    # and sm contains test names like "test/test_InfoExtractor.py::TestInfoExtractor::test_download_json"
    matching_keys = [k for k in sm if k.startswith(case + '::')]

    if len(matching_keys) == 0:
        return False

    if all(sm[k] in [TestStatus.PASSED.value, TestStatus.XFAIL.value] for k in matching_keys):
        return True

    return False


def test_failed_prefix_match(case: str, sm: dict[str, str]) -> bool:
    if case in sm:
        return sm[case] in [TestStatus.FAILED.value, TestStatus.ERROR.value]

    # alternate version which can handle case where "case" is something like "test/test_InfoExtractor"
    # and sm contains test names like "test/test_InfoExtractor.py::TestInfoExtractor::test_download_json"
    matching_keys = [k for k in sm if k.startswith(case + '::')]

    if len(matching_keys) == 0:
        # this seems wrong to me idk. but just mirroring logic from test_failed function.
        return True

    if any(sm[k] in [TestStatus.FAILED.value, TestStatus.ERROR.value] for k in matching_keys):
        return True

    return False


# MARK: Utility functions
def test_passed(case: str, sm: dict[str, str]) -> bool:
    return case in sm and sm[case] in [TestStatus.PASSED.value, TestStatus.XFAIL.value]


def test_failed(case: str, sm: dict[str, str]) -> bool:
    return case not in sm or any(
        sm[case] == status for status in [TestStatus.FAILED.value, TestStatus.ERROR.value]
    )


# MARK: Evaluation report functions
def get_logs_eval(log_fp: str) -> tuple[dict[str, str], bool]:
    """
    Retrieve evaluation results for a task instance from its corresponding log file

    Args:
        log_fp (str): path to log file
    Returns:
        bool: whether the patch applied successfully
        dict: status map

    TODO(john-b-yang): Check this is working properly...
    """
    # Convert e.g. "logs/scikit-learn__scikit-learn-12421/test_output.txt" to "scikit-learn/scikit-learn"
    sample_id = str(Path(log_fp).parent.stem)  # e.g. scikit-learn__scikit-learn-12421
    repo = "-".join(sample_id.replace("__", "/").split("-")[:-1])  # e.g. scikit-learn/scikit-learn
    log_parser = MAP_REPO_TO_PARSER[repo]

    with open(log_fp) as f:
        content = f.read()
        # TODO fix constant here
        if (
            any(
                [
                    x in content
                    for x in [
                        APPLY_PATCH_FAIL,
                        RESET_FAILED,
                        TESTS_ERROR,
                        TESTS_TIMEOUT,
                        "Failed to reset task environment",
                    ]
                ]
            )
            # TODO: find a workaround for this
            #or "applied patch" not in content.lower() --> Base commit is used to simulate bad patch for now
        ):
            # Eval patch was not applied successfully
            return {}, False

        # Get status map of evaluation results
        content = content.split(f"{APPLY_PATCH_PASS} (pred)")[-1]
        return log_parser(content), True


def get_eval_tests_report(
    eval_sm: dict[str, str],
    gold_results: dict[str, str],
    calculate_to_fail: bool = False,
) -> dict[str, dict[str, list[str]]]:
    """
    Create a report based on failure/pass change from gold results to eval results.

    Args:
        eval_sm (dict): evaluation status map
        gold_results (dict): gold results
        calculate_to_fail (bool): whether to calculate metrics for "x to fail" tests
    Returns:
        report (dict): report of metrics

    Metric Definitions (Gold Result Pair + Eval Result):
    - Fail-Pass (F2P) + P: Success (Resolution)
    - Pass-Pass (P2P) + P: Success (Maintenance)
    - Fail-Pass (F2P) + F: Failure
    - Pass-Pass (P2P) + F: Failure

    Miscellaneous Definitions
    - Fail-Fail (F2F) + F: Failure Maintenance
    - Pass-Fail (P2F) + F: Not considered
    - Fail-Fail (F2F) + P: Success (Extra Credit)
    - Pass-Fail (P2F) + P: Not considered
    """
    # Calculate resolution metrics
    f2p_success = []
    f2p_failure = []
    for test_case in gold_results[FAIL_TO_PASS]:
        if test_passed(test_case, eval_sm):
            # Assume silent success for now (test case not in eval_sm)
            f2p_success.append(test_case)
        elif test_failed(test_case, eval_sm):
            f2p_failure.append(test_case)

    # Calculate maintenance metrics
    p2p_success = []
    p2p_failure = []
    for test_case in gold_results[PASS_TO_PASS]:
        if test_passed(test_case, eval_sm):
            p2p_success.append(test_case)
        elif test_failed(test_case, eval_sm):
            p2p_failure.append(test_case)

    results = {
        FAIL_TO_PASS: {
            "success": f2p_success,
            "failure": f2p_failure,
        },
        PASS_TO_PASS: {
            "success": p2p_success,
            "failure": p2p_failure,
        },
    }

    f2f_success = []
    f2f_failure = []
    p2f_success = []
    p2f_failure = []
    if calculate_to_fail:
        # Calculate "extra credit" metrics
        for test_case in gold_results[FAIL_TO_FAIL]:
            if test_passed(test_case, eval_sm):
                f2f_success.append(test_case)
            elif test_failed(test_case, eval_sm):
                f2f_failure.append(test_case)

        # Calculate not considered metrics
        for test_case in gold_results[PASS_TO_FAIL]:
            if test_passed(test_case, eval_sm):
                p2f_success.append(test_case)
            elif test_failed(test_case, eval_sm):
                p2f_failure.append(test_case)

    results.update(
        {
            FAIL_TO_FAIL: {
                "success": f2f_success,
                "failure": f2f_failure,
            },
            PASS_TO_FAIL: {
                "success": p2f_success,
                "failure": p2f_failure,
            },
        }
    )
    return results


def compute_fail_to_pass(report: dict[str, dict[str, Any]]) -> float:
    """
    Compute fail-to-pass metric. Accepts single report as argument.
    """
    total = len(report[FAIL_TO_PASS]["success"]) + len(report[FAIL_TO_PASS]["failure"])
    if total == 0:
        return 1
    return len(report[FAIL_TO_PASS]["success"]) / total


def compute_pass_to_pass(report: dict[str, dict[str, Any]]) -> float:
    """
    Compute pass-to-pass metric. Accepts single report as argument.
    """
    total = len(report[PASS_TO_PASS]["success"]) + len(report[PASS_TO_PASS]["failure"])
    if total == 0:
        # TODO: Don't factor in p2p metrics
        return 1
    return len(report[PASS_TO_PASS]["success"]) / total


def get_resolution_status(report: dict[str, dict[str, Any]]) -> str:
    """
    Determine resolved status of an evaluation instance

    Criteria:
        - If fail-to-pass (Resolution) = 1 and pass-to-pass (Maintenance) = 1 -> FULL
        - If (fail-to-pass (Resolution) < 1 and > 0) and pass-to-pass (Maintenance) = 1 -> PARTIAL
        - Otherwise -> NO
    """
    f2p = compute_fail_to_pass(report)
    p2p = compute_pass_to_pass(report)

    if f2p == 1 and p2p == 1:
        return ResolvedStatus.FULL.value
    elif f2p < 1 and f2p > 0 and p2p == 1:
        return ResolvedStatus.PARTIAL.value
    else:
        return ResolvedStatus.NO.value

def evaluate_report_TestGeneration(report: dict[str, dict[str, Any]]) -> str:
        # Extract values from the report
    expected_pass = report.get("EXPECTED_PASS", {})
    expected_fail = report.get("EXPECTED_FAIL", {})

    # Calculate success and failure counts for both pass (GOLD) and fail (bad patch) categories
    pass_success = len(expected_pass.get("success", []))
    pass_failure = len(expected_pass.get("failure", []))

    expected_fail = merge_and_unpack(expected=expected_fail)
    fail_success = len(expected_fail.get("success", []))
    fail_failure = len(expected_fail.get("failure", []))

    # Evaluate the combination of pass and fail categories
    # TODO: Re-Evaluate this, is it reasonable to assume full test file will pass for Gold?
    if pass_success > 0 and pass_failure == 0:
        if fail_success > 0 and fail_failure >= 0:
            return TestedStatus.SUCCESS  # Full success in pass (GOLD) and fail (bad)
        else:
            return TestedStatus.SUCCESS_GOLD  # Only success in pass (GOLD)
    else:
        if fail_success > 0:
            return TestedStatus.SUCCESS_BAD  # Partial success in pass (GOLD)
        else:
            return TestedStatus.FAIL  # Partial success in pass (GOLD)




def get_eval_report(
    test_spec: TestSpec,
    prediction: dict[str, str],
    log_path: str,
    include_tests_status: bool,
) -> dict[str, Any]:
    """
    Generate a report of model evaluation results from a prediction, task instance,
    and evaluation log.

    Args:
        test_spec (dict): test spec containing keys "instance_id", "FAIL_TO_PASS", and "PASS_TO_PASS"
        prediction (dict): prediction containing keys "instance_id", "model_name_or_path", and "model_patch"
        log_path (str): path to evaluation log
        include_tests_status (bool): whether to include the status of each test in the returned report
    Returns:
        report (dict): report of metrics
    """
    report_map = {}

    instance_id = prediction[KEY_INSTANCE_ID]
    report_map[instance_id] = {
        "patch_is_None": False,
        "patch_exists": False,
        "patch_successfully_applied": False,
        "resolved": False,
    }

    # Check if the model patch exists
    if prediction["model_patch"] is None:
        report_map[instance_id]["patch_is_None"] = True
        return report_map
    report_map[instance_id]["patch_exists"] = True

    # Get evaluation logs
    eval_sm, found = get_logs_eval(log_path)

    if not found:
        return report_map
    report_map[instance_id]["patch_successfully_applied"] = True

    eval_ref = {
        KEY_INSTANCE_ID: test_spec.instance_id,
        FAIL_TO_PASS: test_spec.FAIL_TO_PASS,
        PASS_TO_PASS: test_spec.PASS_TO_PASS,
    }

    report = get_eval_tests_report(eval_sm, eval_ref)
    if get_resolution_status(report) == ResolvedStatus.FULL.value:
        report_map[instance_id]["resolved"] = True

    if include_tests_status:
        report_map[instance_id]["tests_status"] = report  # type: ignore

    return report_map

def get_eval_tests_report_TestGeneration(
    eval_sm: dict[str, str],
    is_gold_patch: bool = False,
    calculate_to_fail: bool = False,
    fail_to_fail: list[str] = []
) -> dict[str, dict[str, list[str]]]:
    """
    Create a report based on failure/pass change given a gold or bad patch using candidate test.

    Args:
        eval_sm (dict): evaluation status map
        is_gold_patch (bool): whether applied issue patch is gold
        calculate_to_fail (bool): whether to calculate metrics for "x to fail" tests
    Returns:
        report (dict): report of metrics

    """

    # TODO: When supported, account for F2F metrics (e.g disregard for evaluation further down)
    successes = []
    failures = []
    for test_case in eval_sm: # Hopefully iterates over all.
        if test_passed(test_case, eval_sm):
            # Assume silent success for now (test case not in eval_sm)
            if(is_gold_patch):
                successes.append(test_case)
            else:
                failures.append(test_case)
        elif test_failed(test_case, eval_sm):
            if(is_gold_patch) and not test_case in fail_to_fail:
                failures.append(test_case)
            else:
                successes.append(test_case)
    if(is_gold_patch):
        results = {
            "EXPECTED_PASS": {
                "success": successes,
                "failure": failures,
            },
        }
    else:
        results = {
            "EXPECTED_FAIL": {
                "success": successes,
                "failure": failures,
            }
        }

    return results

def get_eval_report_test_generation(
    test_spec: TestSpec,
    prediction: dict[str, str],
    log_paths: list[str],
    include_tests_status: bool,
    fail_to_fail_tests: list[str] = []
) -> dict[str, Any]:
    """
    Generate a report of model evaluation results from a prediction, task instance,
    and evaluation logs.

    Args:
        test_spec (dict): test spec containing keys "instance_id", "FAIL_TO_PASS", and "PASS_TO_PASS"
        prediction (dict): prediction containing keys "instance_id", "model_name_or_path", and "model_patch"
        log_paths (str): path to evaluation logs, first is expected to be gold patch, rest are bad patches
        include_tests_status (bool): whether to include the status of each test in the returned report
    Returns:
        report (dict): report of metrics
    """
    report_map = {}

    instance_id = prediction[KEY_INSTANCE_ID]
    report_map[instance_id] = {
        "patch_is_None": False,
        "patch_exists": False,
        "gold_patch_successfully_applied": False,
        "bad_patches_results": [],
        "Test_Accept": False
    }

    # Check if the model patch exists
    if prediction.get("candidate_test_patch") is None and prediction.get("model_patch") is None:
        report_map[instance_id]["patch_is_None"] = True
        return report_map

    report_map[instance_id]["patch_exists"] = True

    # Get evaluation logs for gold patch
    eval_sm_gold, found_gold = get_logs_eval(log_paths[0])
    if not found_gold:
        return report_map

    report_map[instance_id]["gold_patch_successfully_applied"] = True
    report_gold = get_eval_tests_report_TestGeneration(eval_sm_gold, is_gold_patch=True)

    # Process each bad patch result
    for i, bad_patch_log in enumerate(log_paths[1:]):
        eval_sm_bad, found_bad = get_logs_eval(bad_patch_log)
        
        bad_patch_result = {
            "patch_index": i,
            "successfully_applied": found_bad,
            "tests_status": None
        }

        if found_bad:
            report_bad = get_eval_tests_report_TestGeneration(eval_sm_bad, is_gold_patch=False)
            bad_patch_result["tests_status"] = report_bad

        report_map[instance_id]["bad_patches_results"].append(bad_patch_result)

    # A test is accepted if:
    # 1. Gold patch tests pass
    # 2. At least one bad patch fails the tests
    gold_tests_pass = (
        len(report_gold.get("EXPECTED_PASS", {}).get("success", [])) > 0 and
        len(report_gold.get("EXPECTED_PASS", {}).get("failure", [])) == 0
    )

    # Check if any bad patch failed the tests (which is what we want)
    any_bad_patch_failed = False
    for result in report_map[instance_id]["bad_patches_results"]:
        if result["tests_status"]:
            bad_tests = result["tests_status"].get("EXPECTED_FAIL", {})
            if bad_tests.get("success", []):  # If there are failures, that's good!
                any_bad_patch_failed = True
                break

    report_map[instance_id]["Test_Accept"] = gold_tests_pass and any_bad_patch_failed

    if include_tests_status:
        report_map[instance_id]["gold_tests_status"] = report_gold

    return report_map

def get_fail_to_fail(
        gold_log_path: str
) -> list[str]:
    """
    Gets the failed test case names from the current evaluation log.

    Args:
      gold_log_path (str) : Path to the gold patch evaluation log.
    Returns:
      fail_to_fail_tests (list[str]) : Names of F2F tests.

    """

    fail_to_fail_tests = []

    eval_sm_gold, found_gold = get_logs_eval(gold_log_path)

    if not found_gold:
            return []

    for test_case in eval_sm_gold: # Hopefully iterates over all.
        if test_failed(test_case, eval_sm_gold):
                fail_to_fail_tests.append(test_case)

    return fail_to_fail_tests
