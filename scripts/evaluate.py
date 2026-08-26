#!/usr/bin/env python3
"""Stage-gated entry point: Evaluate a baseline or later model."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a baseline or later model.")
    parser.parse_args()
    parser.exit(2, "not implemented until Phase 1\n")


if __name__ == "__main__":
    main()
