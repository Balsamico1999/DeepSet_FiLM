"""Convenience entry point; see python generate_synthetic_data.py --help."""

import sys
from fpga_bench.__main__ import main

if __name__ == "__main__":
    raise SystemExit(main(["generate", *sys.argv[1:]]))
