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
"keras-team/keras": {
    "MAP_REPO_TO_VERSION_PATHS": ["keras/src/version.py"],
    "MAP_REPO_TO_VERSION_PATTERNS":  ["__version__ = ['\"](.*)['\"]", "VERSION = \\((.*)\\)"],
    "MAP_REPO_VERSION_TO_SPECS": {
        k: {
            "python": "3.10",
            "packges": "requirements.txt",
            # "install": "python pip_build.py --install",
            "install": "python -m pip install -e '.[all, dev, test]'",
            "test_cmd": "pytest -rA",
        }
        for k in ['3.3', '3.4', '3.5', '3.6', '3.7', '3.8', None]
    },
    # "MAP_REPO_TO_REQS_PATHS": ["requirements.txt"],
},

}
