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
        for k in ['0.55', '0.56']
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
"statsmodels/statsmodels": {
    "MAP_REPO_TO_VERSION_PATHS": ["statsmodels/__init__.py"],
    "MAP_REPO_TO_VERSION_PATTERNS":  ["__version__ = ['\"](.*)['\"]", "VERSION = \\((.*)\\)"],
    "MAP_REPO_VERSION_TO_SPECS": {
            k: {
            "python": "3.9",
            "install": "pip install -e '.[all, dev, test]'",
            "pip_packages": [
                "setuptools==69.0.2",
                "cython==3.0.10",
                "numpy==1.24.3",
                "scipy==1.10.1",
                "setuptools_scm[toml]==8.0.4",
                "pandas==1.5.3",
                "patsy==0.5.3",
                "pytest==7.4.0",
                "pytest-cov==4.1.0",
                "matplotlib==3.7.2"
            ],
            "test_cmd": "pytest -rA --tb=long",
        }
        for k in [None]
    }
},
}
