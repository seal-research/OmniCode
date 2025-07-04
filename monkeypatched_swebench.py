import swebench
import importlib
from pathlib import Path

CUR_DIR = Path(__file__).parent
REPO_DATA_PATH = CUR_DIR / "data/codearena_repo_data.py"
REPO_DATA = eval(REPO_DATA_PATH.read_text())
from CodeArena_grading import test_passed_prefix_match, test_failed_prefix_match

def monkey_patch_swebench():
    # Update constants in swebench
    for instance_repo in REPO_DATA:
        swebench.versioning.constants.MAP_REPO_TO_VERSION_PATHS[instance_repo] = REPO_DATA[instance_repo]["MAP_REPO_TO_VERSION_PATHS"]
        swebench.versioning.constants.MAP_REPO_TO_VERSION_PATTERNS[instance_repo] = REPO_DATA[instance_repo]["MAP_REPO_TO_VERSION_PATTERNS"]
        swebench.harness.constants.MAP_REPO_VERSION_TO_SPECS[instance_repo] = REPO_DATA[instance_repo]["MAP_REPO_VERSION_TO_SPECS"]

        from swebench.harness.log_parsers.python import parse_log_pytest
        if "MAP_REPO_TO_PARSER" in REPO_DATA[instance_repo]:
            repo_log_parser = eval(REPO_DATA[instance_repo]["MAP_REPO_TO_PARSER"])
        else:
            repo_log_parser = parse_log_pytest
        swebench.harness.log_parsers.MAP_REPO_TO_PARSER[instance_repo] = repo_log_parser

        if "MAP_REPO_TO_REQS_PATHS" in REPO_DATA[instance_repo]:
            swebench.harness.constants.MAP_REPO_TO_REQS_PATHS[instance_repo] = REPO_DATA[instance_repo]["MAP_REPO_TO_REQS_PATHS"]

    # monkey patch the test_passed and test_failed functions in grading.py
    swebench.harness.grading.test_passed = test_passed_prefix_match
    swebench.harness.grading.test_failed = test_failed_prefix_match

    importlib.reload(swebench)

monkey_patch_swebench()
