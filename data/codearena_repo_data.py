{
    "scrapy/scrapy": {
        "MAP_REPO_TO_VERSION_PATHS": ["scrapy/VERSION"],
        "MAP_REPO_TO_VERSION_PATTERNS": ["(\S*)"],
        "MAP_REPO_VERSION_TO_SPECS": {
            **{
                k: {
                    "python": "3.9",
                    "install": "pip install -e '.'",
                    "pip_packages": [
                        "tox",
                        "testfixtures",
                        "pytest",
                        "pexpect",
                        "botocore",
                        "boto3",
                        "google-cloud-storage",
                    ],
                    "test_cmd": "pytest -rA --tb=long",
                }
                for k in ["2.6", "2.7", "2.8", "2.9"]
            },
            **{
                k: {
                    "python": "3.9",
                    "install": "pip install -e '.'",
                    "pip_packages": [
                        "tox",
                        "testfixtures",
                        "pytest",
                        "pexpect",
                    ],
                    "test_cmd": "pytest -rA --tb=long",
                }
                for k in ["2.12", "2.11", "2.10"]
            }
        },
    },
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
}
