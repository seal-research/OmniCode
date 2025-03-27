{
"camel-ai/camel": {
    "MAP_REPO_TO_VERSION_PATHS": ["camel/__init__.py"],
    "MAP_REPO_TO_VERSION_PATTERNS": ["__version__ = ['\"](.*)['\"]", "VERSION = \\((.*)\\)"],
    "MAP_REPO_VERSION_TO_SPECS": {
            k: {
            "python": "3.10",
            "install": "pip install -e '.[all, dev, test]'",
            "pip_packages" : ["'gradio'"],
            "test_cmd": "pytest -rA --tb=long",
        }
        for k in ["0.2"]
    }
}
}
