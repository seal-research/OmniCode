{
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