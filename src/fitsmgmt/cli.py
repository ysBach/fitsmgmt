import argparse
import sys
from .filemgmt import make_summary
from .logging import enable_console_logging
import logging

def run_summary():
    parser = argparse.ArgumentParser(description="Create a summary table from FITS files.")
    parser.add_argument("inputs", nargs="+", help="Input FITS files (glob patterns allowed)")
    parser.add_argument("-o", "--output", help="Output file (CSV, etc.)", default=None)
    parser.add_argument("-k", "--keywords", nargs="+", help="Keywords to extract", default=None)
    parser.add_argument("-e", "--extension", help="Extension to read", default=None)
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if args.verbose:
        enable_console_logging(level=logging.INFO)

    try:
        df = make_summary(
            inputs=args.inputs,
            extension=args.extension,
            keywords=args.keywords,
            output=args.output,
            verbose=args.verbose
        )
        if df is not None and args.output is None:
            print(df)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    run_summary()
