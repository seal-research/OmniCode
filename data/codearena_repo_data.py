{
"Avaiga/taipy": {
    "MAP_REPO_TO_VERSION_PATHS": ["taipy/__init__.py"],
    "MAP_REPO_TO_VERSION_PATTERNS":  ["__version__ = ['\"](.*)['\"]", "VERSION = \\((.*)\\)"],
    "MAP_REPO_VERSION_TO_SPECS": {
            k: {
            "python": "3.9",
            "install": "pip install -e .",
            "pip_packages": [
            ],
            "test_cmd": "pytest -rA --tb=long", 
        }
        for k in ['4.0']
    }
},
}