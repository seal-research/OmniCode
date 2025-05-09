import json
from pathlib import Path
import numpy as np

def clean(
    data: str | Path,
):
    data_path = Path(data)
    new_data = []

    for record in json.loads(data_path.read_text()):
        for k in ["FAIL_TO_PASS", "PASS_TO_PASS"]:
            if k in record:
                if isinstance(record[k], str):
                    p = json.loads(record[k])
                else:
                    p = record[k]

                if isinstance(p, float) and np.isnan(p):
                    p = []

                record[k] = p if p is not None else []

        new_data.append(record)

    data_path.write_text(json.dumps(new_data, indent=2))

def check(
    data: str | Path
):
    problematic_records = []
    
    for idx, record in enumerate(json.loads(Path(data).read_text())):
        if "PASS_TO_PASS" in record:
            value = record["PASS_TO_PASS"]
            if not isinstance(value, (str, list)) and value is not None:
                problematic_records.append({
                    "index": idx,
                    "value_type": type(value).__name__,
                    "value": value,
                    "record_preview": {k: str(v)[:30] + "..." if isinstance(v, str) and len(str(v)) > 30 else v 
                                       for k, v in list(record.items())[:5]}
                })
    
    print(f"Found {len(problematic_records)} problematic records")
    for i, rec in enumerate(problematic_records[:5]):  # Show first 5 problematic records
        print(f"Problem #{i+1}:")
        print(f"  Index: {rec['index']}")
        print(f"  PASS_TO_PASS type: {rec['value_type']}")
        print(f"  PASS_TO_PASS value: {rec['value']}")
        print(f"  Record preview: {rec['record_preview']}")
        print()

if __name__ == '__main__':
    import fire

    fire.Fire(check)