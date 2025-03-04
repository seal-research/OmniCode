{
"fastapi/fastapi": {
    "MAP_REPO_TO_VERSION_PATHS": ["fastapi/__init__.py"],
    "MAP_REPO_TO_VERSION_PATTERNS":  ["__version__ = ['\"](.*)['\"]", "VERSION = \\((.*)\\)"],
    "MAP_REPO_VERSION_TO_SPECS": {
            k: {
            "python": "3.7",
            "install": "pip install -e '.[all, dev, test]'",
            "pip_packages": [
                "'flask<2.3.0'"
            ],
            "test_cmd": "pytest -rA --tb=long",
        }
        for k in ['0.55', '0.56', '0.87']
    }
},
"ytdl-org/youtube-dl": {
    "MAP_REPO_TO_VERSION_PATHS": ["youtube_dl/version.py"],
    "MAP_REPO_TO_VERSION_PATTERNS":  ["__version__ = ['\"](.*)['\"]", "VERSION = \\((.*)\\)"],
    "MAP_REPO_VERSION_TO_SPECS": {
            k: {
            "python": "3.7",
            "install": "pip install -e '.[all, dev, test]'",
            "pip_packages": [
                "pytest"
            ],
            "test_cmd": "pytest -rA --tb=long",
        }
        for k in ['2021.12', '2019.11']
    }
},
}
