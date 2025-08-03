from lxml import etree as ET
import json
import sys
import os

def extract_xml_block(raw_text):
    split_marker = "==== FULL CHECKSTYLE VIOLATION XML OUTPUT ===="
    if split_marker not in raw_text:
        raise ValueError("Expected XML split marker not found.")
    return raw_text.split(split_marker, 1)[-1].strip()

def parse_general_xml_report(xml_text):
    results = []
    try:
        parser = ET.XMLParser(recover=True)
        root = ET.fromstring(xml_text.encode("utf-8"), parser=parser)
    except ET.XMLSyntaxError as e:
        raise RuntimeError(f"Failed to parse XML even in recovery mode: {e}")

    for file_elem in root.findall("file"):
        file_path = file_elem.get("name")
        messages = []

        for err in file_elem.findall("error"):
            try:
                messages.append({
                    "line": int(err.get("line")),
                    "column": int(err.get("column")) if err.get("column") else 0,
                    "type": err.get("severity"),
                    "message": err.get("message"),
                    "source": err.get("source")
                })
            except Exception:
                continue  # skip malformed entries

        error_count = len(messages)
        score = max(0, 10 - error_count / 2.0)

        results.append({
            "file": file_path,
            "score": round(score, 2),
            "error_count": error_count,
            "messages": messages
        })

    return results

def process_log(dir_path, log_filename, output_filename):
    input_path = os.path.join(dir_path, log_filename)
    output_path = os.path.join(dir_path, output_filename)

    if not os.path.exists(input_path):
        print(f"Warning: File not found: {input_path} (Skipping)")
        return

    with open(input_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    try:
        xml_text = extract_xml_block(raw_text)
        parsed = parse_general_xml_report(xml_text)

        with open(output_path, "w", encoding="utf-8") as out_f:
            json.dump(parsed, out_f, indent=2)

        print(f"Output written to {output_path}")
    except Exception as e:
        print(f"Error while processing {log_filename}: {e}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python log_parser.py <directory_path>")
        sys.exit(1)

    dir_path = sys.argv[1]

    process_log(dir_path, "original_run.log", "original_style_errors.json")
    process_log(dir_path, "patched_run.log", "patched_style_errors.json")

if __name__ == "__main__":
    main()

