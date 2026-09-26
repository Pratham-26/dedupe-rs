#!/usr/bin/env python
"""Run the Python and Rust pipelines across dataset sizes and core counts,
recording wall-clock time and peak resident memory for each.

    python bench_all.py [sizes...] [--cores 1,8]

The Rust `profile` example must have been built:
    (cd ../dedupe-rs && cargo build --release --example profile)
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RS_DIR = os.path.join(HERE, "..", "dedupe-rs")
RS_BIN = os.path.join(RS_DIR, "target", "release", "examples", "profile.exe")
if not os.path.exists(RS_BIN):
    RS_BIN = os.path.join(RS_DIR, "target", "release", "examples", "profile")


def run(cmd, cwd=None):
    out = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"{cmd} failed:\n{out.stdout}\n{out.stderr}")
    last = [ln for ln in out.stdout.strip().splitlines() if ln.strip().startswith("{")]
    return json.loads(last[-1])


def main():
    argv = sys.argv[1:]
    cores_list = [1, 8]
    if "--cores" in argv:
        i = argv.index("--cores")
        cores_list = [int(c) for c in argv[i + 1].split(",")]
        del argv[i : i + 2]
    sizes = [int(a) for a in argv if a.isdigit()] or [2000, 4000, 8000]

    results = []
    for n in sizes:
        print(f"generating dataset n={n} ...", file=sys.stderr)
        subprocess.run(
            [sys.executable, "profile.py", "gen", "1", str(n)],
            cwd=HERE,
            check=True,
            capture_output=True,
        )
        for cores in cores_list:
            print(f"  python n={n} cores={cores} ...", file=sys.stderr)
            py = run([sys.executable, "profile.py", "bench", str(cores), str(n), "--json"], cwd=HERE)
            print(f"  rust   n={n} cores={cores} ...", file=sys.stderr)
            t = time.perf_counter()
            rs = run([RS_BIN, str(cores), "--json"])
            _ = time.perf_counter() - t
            results.append(py)
            results.append(rs)

    rows = []
    for r in results:
        rows.append(
            "| {impl} | {records} | {cores} | {pairs} | {index_all:.2f} | {blocking:.2f} | "
            "{score:.2f} | {cluster:.2f} | {total:.2f} | {peak_mb:.0f} |".format(**r)
        )

    header = (
        "| Impl | Records | Cores | Pairs | index (s) | blocking (s) | score (s) | "
        "cluster (s) | total (s) | peak RSS (MB) |"
    )
    sep = "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    md = "\n".join([header, sep] + rows)
    print(md)

    # Speed-up summary.
    print("\n### Rust speed-up vs Python (same cores)")
    print("| Records | Cores | Py total | Rs total | Speed-up | Py peak | Rs peak | Mem ratio |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for n in sizes:
        for cores in cores_list:
            py = next(r for r in results if r["impl"] == "python" and r["records"] == n and r["cores"] == cores)
            rs = next(r for r in results if r["impl"] == "rust" and r["records"] == n and r["cores"] == cores)
            print(
                f"| {n} | {cores} | {py['total']:.2f} | {rs['total']:.2f} | "
                f"{py['total'] / rs['total']:.2f}x | {py['peak_mb']:.0f} | {rs['peak_mb']:.0f} | "
                f"{py['peak_mb'] / rs['peak_mb']:.2f}x |"
            )


if __name__ == "__main__":
    main()
