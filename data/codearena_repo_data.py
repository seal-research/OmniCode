{
"sherlock-project/sherlock": {
    "MAP_REPO_TO_VERSION_PATHS": ["sherlock_project/__init__.py"],
    "MAP_REPO_TO_VERSION_PATTERNS":  ["__version__ = ['\"](.*)['\"]", "VERSION = \\((.*)\\)"],
    "MAP_REPO_VERSION_TO_SPECS": {
            k: {
            "python": "3.7",
            "install": "pip install -e .",
            "pip_packages": [],
            "test_cmd": "pytest -rA --tb=long", 
        }
        for k in [None]
    }
},
}