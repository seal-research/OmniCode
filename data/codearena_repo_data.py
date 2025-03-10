{
"celery/celery": {
    "MAP_REPO_TO_VERSION_PATHS": ["celery/__init__.py"],
    "MAP_REPO_TO_VERSION_PATTERNS": ["__version__ = ['\"](.*)['\"]"],
    "MAP_REPO_VERSION_TO_SPECS": {
            k: {
            "python": "3.9", 
            "install": "pip install -U -r requirements/test.txt",
            "pip_packages": [
                "pytest==8.3.4",
                "pytest-xdist",
                "pytest-timeout",
                "pytest-subtests",
                "redis",
                "kombu",
                "vine",
                "amqp",
                "case",
                "billiard",
                "nose",
                "importlib-metadata",
                "importlib-resources",
                "pydantic",
                "dnspython",
                "pymongo>=3.12.0",
                "django-celery",
                "elasticsearch"
            ],
            "test_cmd": "pytest -rA --tb=long"
        }
        for k in ["4.2", "4.3", "4.4","5.0", "5.1", "5.2", "5.3", "5.4", "5.5"]
    }
}
}