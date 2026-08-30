#!/usr/bin/env python3
"""只读汇总 TTS SQLite 状态，供恢复前诊断。"""

from __future__ import annotations

import argparse
import collections
import json
import sqlite3
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--examples", type=int, default=30)
    args = parser.parse_args()
    connection = sqlite3.connect(args.db)
    status = connection.execute("SELECT status, COUNT(*) FROM units GROUP BY status ORDER BY status").fetchall()
    attempts = connection.execute("SELECT attempts, COUNT(*) FROM units GROUP BY attempts ORDER BY attempts").fetchall()
    examples = connection.execute(
        "SELECT key, status, attempts, last_error FROM units WHERE status != 'succeeded' ORDER BY attempts DESC, key LIMIT ?",
        (args.examples,),
    ).fetchall()
    error_counts = connection.execute(
        "SELECT COALESCE(last_error, '<null>'), COUNT(*) FROM units WHERE status != 'succeeded' GROUP BY last_error ORDER BY COUNT(*) DESC LIMIT ?",
        (args.examples,),
    ).fetchall()
    report = {
        "status_counts": dict(status),
        "attempt_counts": dict(attempts),
        "examples": [dict(key=row[0], status=row[1], attempts=row[2], last_error=row[3]) for row in examples],
        "error_counts": [dict(last_error=row[0], count=row[1]) for row in error_counts],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.output:
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
