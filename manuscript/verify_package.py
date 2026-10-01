#!/usr/bin/env python3
"""Check figure paths, Python syntax and table headers."""
import ast
import csv
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parent


def inside(relative):
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Path must be relative to the package: {relative}")
    return ROOT / path


def main():
    entries = json.loads((ROOT / "figures.json").read_text())
    outputs = set()
    for identifier, entry in entries.items():
        files = [entry["script"], *entry["inputs"]]
        for output in [entry["output"], *entry.get("extra_outputs", [])]:
            inside(output)
        if "prepare" in entry:
            files += [entry["prepare"]["script"], entry["prepare"]["output"]]
        for relative in files:
            if not inside(relative).is_file():
                raise FileNotFoundError(f"Figure {identifier}: {relative}")
        if entry["output"] in outputs:
            raise ValueError(f'Duplicate primary output: {entry["output"]}')
        outputs.add(entry["output"])
    forbidden = {
        ".bed",
        ".bim",
        ".fam",
        ".pgen",
        ".pvar",
        ".psam",
        ".bgen",
        ".vcf",
        ".fastq",
        ".bam",
        ".cram",
    }
    identifiers = {"eid", "iid", "fid", "participant_id", "sample_id", "subject_id"}
    tables = 0
    workbooks = 0
    for path in ROOT.rglob("*"):
        if path.relative_to(ROOT).parts[0] in {"output", "inputs"} or any(
            part.startswith(".") or part == "__pycache__"
            for part in path.relative_to(ROOT).parts
        ):
            continue
        if path.is_symlink():
            raise ValueError(f"Unexpected symlink: {path.relative_to(ROOT)}")
        if not path.is_file():
            continue
        if forbidden.intersection(path.suffixes):
            raise ValueError(f"Unexpected data format: {path.relative_to(ROOT)}")
        if path.suffix == ".py":
            ast.parse(path.read_text(), filename=str(path.relative_to(ROOT)))
        if path.suffix in {".csv", ".tsv"}:
            with path.open(newline="") as handle:
                reader = csv.reader(
                    handle, delimiter="\t" if path.suffix == ".tsv" else ","
                )
                columns = next(reader)
            if identifiers.intersection(column.lower() for column in columns):
                raise ValueError(
                    f"Unexpected identifier column: {path.relative_to(ROOT)}"
                )
            tables += 1
        if path.suffix == ".xlsx":
            with zipfile.ZipFile(path) as book:
                namespace = {
                    "s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
                }
                shared = []
                if "xl/sharedStrings.xml" in book.namelist():
                    root = ET.fromstring(book.read("xl/sharedStrings.xml"))
                    shared = ["".join(node.itertext()) for node in root]
                for name in book.namelist():
                    if not (
                        name.startswith("xl/worksheets/sheet") and name.endswith(".xml")
                    ):
                        continue
                    root = ET.fromstring(book.read(name))
                    first = root.find("s:sheetData/s:row", namespace)
                    if first is None:
                        continue
                    headers = []
                    for cell in first:
                        value = "".join(cell.itertext())
                        headers.append(
                            shared[int(value)] if cell.get("t") == "s" else value
                        )
                    if identifiers.intersection(value.lower() for value in headers):
                        raise ValueError(
                            f"Unexpected identifier column: {path.relative_to(ROOT)} / {name}"
                        )
            workbooks += 1
    print(
        f"Checked {len(entries)} figure definitions, Python syntax and {tables} table headers."
    )
    print(f"Checked {workbooks} metadata workbooks.")


if __name__ == "__main__":
    main()
