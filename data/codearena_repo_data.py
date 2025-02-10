{
"home-assistant/core": {
    "MAP_REPO_TO_VERSION_PATHS": ["homeassistant/const.py"],
    "MAP_REPO_TO_VERSION_PATTERNS": ["__version__ = ['\"](.*)['\"]", "core-(\\d+\\.\\d+\\.\\d+(?:(?:a|b|dev)\\d+)?)", "(\\d+\\.\\d+\\.\\d+(?:(?:a|b|dev)\\d+)?)"],
    "MAP_REPO_VERSION_TO_SPECS": {
        k: {
            "python": "3.13.0",
            "install": "pip install -e '.'",
            "pip_packages": ["pytest==7.4.3", "freezegun", "pytest-asyncio<0.17.0", "pytest_socket", "requests_mock", "respx", "syrupy"],
            "test_cmd": "pytest -rA --tb=long", 
        }
        for k in ['2025.11']
    }
}
}