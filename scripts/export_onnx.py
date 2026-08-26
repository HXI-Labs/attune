#!/usr/bin/env python3
"""Stage-gated entry point: Export a validated model to ONNX."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a validated model to ONNX.")
    parser.parse_args()
    parser.exit(2, "not implemented until Phase 4\n")


if __name__ == "__main__":
    main()
