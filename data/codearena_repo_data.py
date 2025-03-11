{
    "Avaiga/taipy": {
        "MAP_REPO_TO_VERSION_PATHS": ["taipy/__init__.py"],
        "MAP_REPO_TO_VERSION_PATTERNS": ["__version__ = ['\"](.*)['\"]", "VERSION = \\((.*)\\)"],
        "MAP_REPO_VERSION_TO_SPECS": {
            k: {
                "python": "3.9",
                "install": "pip install -e .",
                "pip_packages": [
                    "pytest",
                    "pyngrok>=5.1,<6.0",
                    "python-magic>=0.4.24,<0.5",
                    "python-magic-bin>=0.4.14,<0.5",
                    "rdp>=0.8",
                    "pyarrow>=16.0.0,<19.0",
                    "pyodbc>=4"
                ],
                "test_cmd": "pytest -rA --tb=long"
            }
            for k in [None]
        }
    }
}
