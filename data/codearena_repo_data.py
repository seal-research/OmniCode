{
"OpenInterpreter/open-interpreter": {
    "MAP_REPO_TO_VERSION_PATHS": ["pyproject.toml"],
    "MAP_REPO_TO_VERSION_PATTERNS": ["version\\s*=\\s*['\"]([0-9]+\\.[0-9]+\\.[0-9]+(?:-rc[0-9]+)?)['\"]"],
    "MAP_REPO_VERSION_TO_SPECS": {
            k: {
            "python": "3.11",
            "install": "pip install -e '.[all, dev, test]'",
            "test_cmd": "pytest -rA --tb=long",
        }
        for k in ["0.1"]
    }
},
}