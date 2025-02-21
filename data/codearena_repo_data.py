{
"ultralytics/ultralytics": {
    "MAP_REPO_TO_VERSION_PATHS": ["ultralytics/__init__.py"],
    "MAP_REPO_TO_VERSION_PATTERNS": ["__version__ = ['\"](.*)['\"]", "VERSION = \\((.*)\\)"],
    "MAP_REPO_VERSION_TO_SPECS": {
            k: {
            "python": "3.9",
            "install": "pip install -e '.[all, dev, test]' opencv-python-headless --no-deps",
            "pip_packages": [
                "'pytest'", "'numpy'", "'psutil'", "'torch'", "'Pillow'", "'matplotlib'", "'pyyaml'",
                "'tqdm'", "'torchvision'", "'requests'"
            ],
            "test_cmd": "pytest -rA --tb=long",
        }
        for k in ["8.2"]
    }
},
}