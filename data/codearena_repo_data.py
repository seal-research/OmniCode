{
    "sherlock-project/sherlock": {
        "MAP_REPO_TO_VERSION_PATHS": ["sherlock_project/__init__.py", "sherlock.py"],
        "MAP_REPO_TO_VERSION_PATTERNS": [
            "__version__ = ['\"](.*)['\"]",
            "VERSION = \\((.*)\\)",
            "__version__ *= ['\"](.*)['\"]",
        ],
        "MAP_REPO_VERSION_TO_SPECS": {
            k: {
                "python": "3.7",
                "install": "pip install -e .",
                "test_cmd": "pytest -rA --tb=long",
            }
            for k in ["0.15", "0.3"]
        },
    },
}
