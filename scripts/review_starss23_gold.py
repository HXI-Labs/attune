#!/usr/bin/env python3
"""Generate a local playable STARSS23 gold-review HTML pack.

Hash-verifies first-60s wavs, writes a gitignored HTML pack, and optionally
serves it so Save appends GoldReviewRecord rows to an append-only JSONL ledger.
STARSS23 activity is not Attune gold. This script never trains or wires a head.
"""

from __future__ import annotations

import argparse
import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from attune.data.gold_review import GoldReviewRecord
from attune.data.starss23_gold_pack import (
    DEFAULT_HTML_RELATIVE_PATH,
    DEFAULT_LEDGER_NAME,
    PACK_RELATIVE_PATH,
    AudioHashMismatchError,
    Starss23ReviewStatusError,
    append_ledger,
    assert_expected_pack_counts,
    default_cache_dirs,
    load_jsonl,
    resolve_pack_audio,
    write_html_pack,
)


def generate(arguments: argparse.Namespace) -> dict:
    pack = load_jsonl(arguments.pack)
    assert_expected_pack_counts(pack)
    cache_dirs = (
        list(arguments.cache_dir)
        if arguments.cache_dir
        else default_cache_dirs(arguments.repo_root)
    )
    if not cache_dirs:
        print("warning: no cache directories found; HTML will omit audio")
    resolved, missing, _ = resolve_pack_audio(pack, cache_dirs)
    summary = write_html_pack(
        pack,
        arguments.output_dir,
        resolved,
        missing,
        copy_audio=not arguments.no_copy_audio,
    )
    print(
        f"Wrote {summary['clips']} clip pages / {summary['laugh_events']} source laughs "
        f"to {summary['index']}"
    )
    print(f"Copied {summary['copied_wavs']} hash-matched wavs into the gitignored pack")
    if missing:
        print(f"Missing wavs ({len(missing)}): " + ", ".join(missing))
        print("Refusing to invent audio. Review pages exist without players.")
    else:
        print("All 49 first-60s hashes matched local wavs.")
    print("Gold gate remains CLOSED.")
    return summary


def serve(output_dir: Path, pack_rows: list[dict], ledger: Path, host: str, port: int) -> None:
    pack_by_id = {row["clip_id"]: row for row in pack_rows}

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(output_dir), **kwargs)

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/append":
                self.send_error(404)
                return
            length = int(self.headers.get("content-length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            try:
                record = GoldReviewRecord.model_validate(payload)
                pack_row = pack_by_id.get(record.clip_id)
                if pack_row is None:
                    raise Starss23ReviewStatusError(f"unknown clip_id {record.clip_id}")
                append_ledger(ledger, record, pack_row)
            except (
                AudioHashMismatchError,
                Starss23ReviewStatusError,
                ValueError,
            ) as error:
                self.send_response(400)
                self.send_header("content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(error)}).encode("utf-8"))
                return
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"ok": True, "ledger": str(ledger)}).encode("utf-8"))

        def log_message(self, format: str, *args: object) -> None:
            print(format % args)

    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"Serving {output_dir} at http://{host}:{port}/index.html")
    print(f"Append-only ledger: {ledger}")
    print("Do not expose this server; STARSS23 audio stays local.")
    httpd.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--pack", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        action="append",
        default=None,
        help="Directory of first-60s wavs; may be repeated. Searched by hash.",
    )
    parser.add_argument(
        "--no-copy-audio",
        action="store_true",
        help="Link original wavs instead of copying into the gitignored pack",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Serve the pack locally so Save appends ledger.jsonl",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    arguments = parser.parse_args()
    if arguments.pack is None:
        arguments.pack = arguments.repo_root / PACK_RELATIVE_PATH
    if arguments.output_dir is None:
        arguments.output_dir = arguments.repo_root / DEFAULT_HTML_RELATIVE_PATH
    generate(arguments)
    if arguments.serve:
        pack_rows = load_jsonl(arguments.pack)
        ledger = arguments.output_dir / DEFAULT_LEDGER_NAME
        serve(arguments.output_dir, pack_rows, ledger, arguments.host, arguments.port)


if __name__ == "__main__":
    main()
