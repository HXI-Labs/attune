#!/usr/bin/env python3
"""Stage-gated entry point: Prepare a reviewed dataset manifest."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description='Prepare a reviewed dataset manifest.')
    parser.parse_args()
    parser.exit(2, "not implemented until Phase 1\n")


if __name__ == "__main__":
    main()
