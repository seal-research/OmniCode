{
"freqtrade/freqtrade": {
    "MAP_REPO_TO_VERSION_PATHS": ["freqtrade/__init__.py"],
    "MAP_REPO_TO_VERSION_PATTERNS": ["__version__ = ['\"](.*)['\"]", "VERSION = \\((.*)\\)"],
    "MAP_REPO_VERSION_TO_SPECS": {
            k: {
            "python": "3.10",
            "install": "pip install python-dateutil --upgrade && conda install -c conda-forge ta-lib && pip install -e '.[all, dev, test]'",
            "pip_packages" : ["'pytest-xdist'"],
            "test_cmd": "pytest -rA --tb=long",
        }
        for k in ["2024.9"]
    }
},
}