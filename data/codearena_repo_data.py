{
"apache/airflow": {
    "MAP_REPO_TO_VERSION_PATHS": ["airflow/__init__.py"],
    "MAP_REPO_TO_VERSION_PATTERNS": ["__version__ = ['\"](.*)['\"]"],
    "MAP_REPO_VERSION_TO_SPECS": {
            k: {
            "python": "3.12", 
            "install": "pip install -e '.[all,dev,test]' --no-deps && pip install --no-deps -r <(pip freeze)",
            "pip_packages": [
                "sqlalchemy",
                "time_machine",
                "fastapi",
                "httpx",
                "pytest-asyncio",
                "methodtools",
                "google-re2",
                "PyYAML",
                "cryptography",
                "pendulum",
                "termcolor"
            ],
            "test_cmd": "pytest -rA --tb=long"
        }
        for k in ["3.0"]
    }
},
}