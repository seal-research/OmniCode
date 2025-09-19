import json
import sys
from pathlib import Path


def main(testfile):
    with open('coverage.json') as f:
        data = json.load(f)
    with open('map.json', 'r') as f:
        map = json.load(f)

    for code_file, info in data.get("files", {}).items():
        fname = Path(code_file).name
        if fname.startswith("test_") or fname.endswith("_test.py"):
            continue
        if info.get("summary").get("percent_covered") > 0:
            if code_file not in map.keys():
                map[code_file] = []
            map[code_file].append(testfile)

    with open('map.json', 'w') as f:
        json.dump(map, f, indent=2)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python build_mapping.py test_file")
        sys.exit(1)
    main(sys.argv[1])