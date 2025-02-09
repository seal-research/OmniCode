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
"scrapy/scrapy": {
    "MAP_REPO_TO_VERSION_PATHS": ["scrapy/VERSION"],
    "MAP_REPO_TO_VERSION_PATTERNS":  ["(\S*)"],
    "MAP_REPO_VERSION_TO_SPECS": {
            k: {
            "python": "3.9",
            "install": "pip install -e '.'",
            "pip_packages": [
                "tox",
                "testfixtures",
                "pytest",
            ],
            "test_cmd": "pytest -rA --tb=long", 
        }
        for k in ['2.12', '2.11']
    }
},
}