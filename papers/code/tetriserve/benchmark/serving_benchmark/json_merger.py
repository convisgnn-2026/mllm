#!/usr/bin/env python3
"""
JSON Merger Script

This script merges multiple JSON files into a single JSON file.
Each input JSON file should contain a list or will be treated as a single item.

Usage:
    python json_merger.py -i file1.json file2.json file3.json -o merged.json
    python json_merger.py --input file1.json file2.json --output merged.json
    python json_merger.py -i *.json -o all_merged.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Any, Union


class JSONMerger:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.merged_data = []
        self.file_count = 0
        self.item_count = 0

    def log(self, message: str):
        """Log message if verbose mode is enabled."""
        if self.verbose:
            print(f"[JSONMerger] {message}")

    def load_json_file(self, file_path: str) -> Union[List[Any], Any]:
        """
        Load JSON data from a file.
        Returns the data as-is, whether it's a list, dict, or other type.
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.log(f"Loaded {file_path}: {type(data).__name__}")
            return data

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in file {file_path}: {e}")
        except Exception as e:
            raise RuntimeError(f"Error reading file {file_path}: {e}")

    def merge_data(self, data: Any) -> None:
        """
        Merge data into the main list.
        If data is a list, extend the main list.
        If data is not a list, append it as a single item.
        """
        if isinstance(data, list):
            self.merged_data.extend(data)
            items_added = len(data)
            self.log(f"Extended with {items_added} items from list")
        else:
            self.merged_data.append(data)
            items_added = 1
            self.log(f"Appended 1 item ({type(data).__name__})")

        self.item_count += items_added

    def merge_files(self, input_files: List[str]) -> None:
        """Merge all input files into the merged_data list."""
        self.log(f"Starting merge of {len(input_files)} files")

        for file_path in input_files:
            try:
                data = self.load_json_file(file_path)
                self.merge_data(data)
                self.file_count += 1
                self.log(f"Successfully processed: {file_path}")

            except Exception as e:
                print(f"Error: Failed to process {file_path}: {e}", file=sys.stderr)
                continue

        self.log(f"Merge complete: {self.file_count} files, {self.item_count} total items")

    def save_output(self, output_file: str, indent: int = 2) -> None:
        """Save merged data to output file."""
        try:
            output_path = Path(output_file)

            # Create parent directories if they don't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(self.merged_data, f, indent=indent, ensure_ascii=False)

            self.log(f"Saved merged data to: {output_path}")
            print(f"✅ Successfully merged {self.file_count} files into {output_path}")
            print(f"📊 Total items in merged file: {self.item_count}")

        except Exception as e:
            raise RuntimeError(f"Error writing output file {output_file}: {e}")

    def get_summary(self) -> dict:
        """Get summary of the merge operation."""
        return {
            "files_processed": self.file_count,
            "total_items": self.item_count,
            "merged_data_type": type(self.merged_data).__name__,
            "output_size": len(self.merged_data),
        }


def validate_input_files(input_files: List[str]) -> List[str]:
    """Validate and expand input file patterns."""
    expanded_files = []

    for pattern in input_files:
        path = Path(pattern)
        if path.is_file():
            expanded_files.append(str(path))
        elif "*" in pattern or "?" in pattern:
            # Handle glob patterns
            matches = list(Path(".").glob(pattern))
            if matches:
                expanded_files.extend(str(p) for p in matches if p.is_file())
            else:
                print(f"Warning: No files found matching pattern: {pattern}", file=sys.stderr)
        else:
            print(f"Warning: File not found: {pattern}", file=sys.stderr)

    return expanded_files


def main():
    parser = argparse.ArgumentParser(
        description="Merge multiple JSON files into a single JSON file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Merge specific files
  python json_merger.py -i file1.json file2.json file3.json -o merged.json

  # Merge all JSON files in current directory
  python json_merger.py -i *.json -o all_data.json

  # Merge with custom formatting
  python json_merger.py -i data1.json data2.json -o output.json --indent 4 --verbose

  # Merge benchmark results
  python json_merger.py -i benchmark_results_*.json -o combined_results.json
        """,
    )

    parser.add_argument(
        "-i", "--input", nargs="+", required=True, help="Input JSON files to merge (supports glob patterns like *.json)"
    )

    parser.add_argument("-o", "--output", required=True, help="Output JSON file path")

    parser.add_argument("--indent", type=int, default=2, help="JSON indentation level (default: 2)")

    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be merged without actually creating output file"
    )

    args = parser.parse_args()

    # Validate and expand input files
    input_files = validate_input_files(args.input)

    if not input_files:
        print("Error: No valid input files found", file=sys.stderr)
        sys.exit(1)

    if args.verbose:
        print(f"Input files to process: {len(input_files)}")
        for f in input_files:
            print(f"  - {f}")
        print()

    # Create merger and process files
    merger = JSONMerger(verbose=args.verbose)

    try:
        merger.merge_files(input_files)

        if args.dry_run:
            print("=" * 50)
            print("DRY RUN MODE")
            print("=" * 50)
            summary = merger.get_summary()
            print(f"Would merge {summary['files_processed']} files")
            print(f"Total items: {summary['total_items']}")
            print(f"Output file: {args.output}")
            print("No files were actually created.")
        else:
            merger.save_output(args.output, indent=args.indent)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
