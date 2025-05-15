{
"fastapi/fastapi": {
    "MAP_REPO_TO_VERSION_PATHS": ["fastapi/__init__.py"],
    "MAP_REPO_TO_VERSION_PATTERNS":  ["__version__ = ['\"](.*)['\"]", "VERSION = \\((.*)\\)"],
    "MAP_REPO_VERSION_TO_SPECS": {
            k: {
            "python": "3.7",
            "install": "pip install -e '.[all, dev, test]'",
            "pip_packages": [
                "'flask<2.3.0'",
                "websockets"
            ],
            "test_cmd": "pytest -rA --tb=long",
        }
        for k in ['0.87', '0.94', '0.68', '0.92', '0.79', '0.79', '0.55', '0.55', '0.57', '0.55', '0.55', '0.55', '0.52', '0.49', '0.47', '0.46', '0.45', '0.42', '0.42', '0.39', '0.35', '0.35', '0.35', '0.35', '0.31', '0.29']
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
            "pip_packages": [
                'namex>=0.0.8',
                'ruff',
                'pytest',
                'numpy',
                'scipy',
                'scikit-learn',
                'pandas',
                'absl-py',
                'requests',
                'h5py',
                'ml-dtypes',
                'protobuf',
                'google',
                'tensorboard-plugin-profile',
                'rich',
                'build',
                'optree',
                'pytest-cov',
                'packaging',
                # for tree_test.py
                'dm_tree',
                'coverage!=7.6.5',  # 7.6.5 breaks CI
                # for onnx_test.py
                'onnxruntime',
                # Tensorflow.
                "tensorflow~=2.18.0",
                'tf_keras',
                'tf2onnx',
                # Torch.
                'torch>=2.1.0',
                'torchvision>=0.16.0',
                'torch-xla',
                # Jax.
                'jax[cpu]',
                'flax',
            ],
            # "install": "python pip_build.py --install",
            "install": "python -m pip install -e .",
            "test_cmd": "pytest -rA",
        }
        for k in ['3.3', '3.4', '3.5', '3.6', '3.7', '3.8', 'None']
    },
    # "MAP_REPO_TO_REQS_PATHS": ["requirements.txt"],
},
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
        for k in ['None']
    }
},
"scrapy/scrapy": {
        "MAP_REPO_TO_VERSION_PATHS": ["scrapy/VERSION"],
        "MAP_REPO_TO_VERSION_PATTERNS": ["(\S*)"],
        "MAP_REPO_VERSION_TO_SPECS": {
            **{
                k: {
                    "python": "3.9",
                    "install": "pip install -e '.'",
                    "pip_packages": [
                        "boto3",
                        "google-cloud-storage",
                        "tox",
                        "testfixtures",
                        "pytest",
                        "sock",
                        "pillow",
                        "botocore",
                        "pexpect",
                        "uvloop",
                        "zstd",
                        "brotli",
                        "twisted",
                    ],
                    "test_cmd": "pytest -rA --tb=long",
                }
                for k in ['2.6', '2.5']
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
                        "botocore",
                        "boto3",
                        "google-cloud-storage",
                    ],
                    "test_cmd": "pytest -rA --tb=long",
                }
                for k in ["2.7", "2.8", "2.9"]
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
                        "sock",
                        "pillow",
                        "botocore",
                        "uvloop",
                        "zstd",
                        "brotli",
                        "twisted",
                    ],
                    "test_cmd": "pytest -rA --tb=long",
                }
                for k in ["2.12", "2.11", "2.10"]
            }
        },
    },
"celery/celery": {
    "MAP_REPO_TO_VERSION_PATHS": ["celery/__init__.py"],
    "MAP_REPO_TO_VERSION_PATTERNS": ["__version__ = ['\"](.*)['\"]"],
    "MAP_REPO_VERSION_TO_SPECS": {

        ** {
            k: {
                "python": "3.9",
                "install": "pip install -U -r requirements/test.txt",
                "pip_packages": [
                    "pytest==8.3.4",
                    "pytest-xdist",
                    "pytest-timeout",
                    "pytest-subtests",
                    "redis",
                    "kombu",
                    "vine",
                    "amqp",
                    "case",
                    "billiard",
                    "nose",
                    "importlib-metadata",
                    "importlib-resources",
                    "pydantic",
                    "dnspython",
                    "pymongo==4.10.1",
                    "django-celery",
                    "elasticsearch",
                    "sqlalchemy==2.0.38",
                    "python-memcached==1.62",
                    "tblib",
                    "pytz",
                    "ephem==4.2.0"
                ],
                "test_cmd": "pytest -rA --tb=long"
            }
            for k in ["5.1", "5.2", "5.3", "5.4", "5.5"]
        },
        **{
            k: {
                "python": "3.9",
                "install": "",
                "pip_packages": [
                    "pytest==7.1.3",
                    "pytest-xdist==2.5.0",
                    "pytest-timeout",
                    "pytest-subtests==0.10.0",
                    "pytest-cov==2.12.1",
                    "case",
                    "kombu",
                    "billiard",
                    "pytz",
                    "click",
                    "flaky",
                    "vine==1.3.0",
                    "redis",
                ],
                "test_cmd": "pytest -rA --tb=long"
            }
            for k in ["5.0"]
        },
        ** {
            k: {
                "python": "3.9",
                "install": "",
                "pip_packages": [
                    "pytest==7.1.3",
                    "pytest-xdist==2.5.0",
                    "pytest-timeout",
                    "pytest-subtests==0.10.0",
                    "pytest-cov==2.12.1",
                    "case",
                    "kombu",
                    "billiard==3.6.4.0",
                    "pytz",
                    "click",
                    "flaky",
                    "vine==1.3.0",
                    "redis",
                    "boto3",
                    "cryptography",
                    "sqlalchemy",
                    "future",
                    "tblib",
                ],
                "test_cmd": "pytest -rA --tb=long"
              }
              for k in ["4.2", "4.3", "4.4"]
        }
    }
  }
}
