from typing import TypedDict

class CodeArenaInstance(TypedDict):
    repo: str
    instance_id: str
    base_commit: str
    patch: str
    test_patch: str
    problem_statement: str
    hints_text: str
    created_at: str
    version: str
    FAIL_TO_PASS: str
    PASS_TO_PASS: str
    environment_setup_commit: str
    bad_patch: str
    candidate_test_patch: str
    bad_patch_author: str
    Review: str
    Review_Author: str