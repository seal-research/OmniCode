{
    "scrapy/scrapy": {
        "MAP_REPO_TO_VERSION_PATHS": ["scrapy/VERSION"],
        "MAP_REPO_TO_VERSION_PATTERNS": ["(\S*)"],
        "MAP_REPO_VERSION_TO_SPECS": {
            k: {
                "python": "3.9",
                "install": "pip install -e '.'",
                "pip_packages": [
                    "tox",
                    "testfixtures",
                    "pytest",
                    "sock",
                    "testbed",
                ],
                "test_cmd": "pytest -rA --tb=long",
            }
            for k in ["2.12", "2.11"]
        },
    },
}
