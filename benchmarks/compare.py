#!/usr/bin/env python3
"""Build isolated v1/v2 runners and compare identical Core ML inference work."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import statistics
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent.parent
RELU = bytes([8, 1, 18, 41, 10, 18, 10, 5, 105, 110, 112, 117, 116, 26, 9, 42, 7, 10, 1, 3, 16, 160, 128, 4, 82, 19, 10, 6, 111, 117, 116, 112, 117, 116, 26, 9, 42, 7, 10, 1, 3, 16, 160, 128, 4, 162, 31, 28, 10, 26, 10, 4, 114, 101, 108, 117, 18, 5, 105, 110, 112, 117, 116, 26, 6, 111, 117, 116, 112, 117, 116, 146, 8, 2, 82, 0])


def command(args, *, cwd=ROOT, env=None, capture=False):
    if not capture:
        print("+", " ".join(str(arg) for arg in args), flush=True)
    return subprocess.run(args, cwd=cwd, env=env, check=True, text=True,
                          stdout=subprocess.PIPE if capture else None).stdout


def export_baseline(revision, destination):
    files = command(["jj", "--ignore-working-copy", "file", "list", "-r", revision], capture=True).splitlines()
    selected = [name for name in files if name in {"Cargo.toml", "Cargo.lock", "build.rs", "LICENSE"} or name.startswith("src/") or name.startswith("swift-library/")]
    if not selected or "Cargo.toml" not in selected:
        raise RuntimeError("baseline does not contain the expected v1 source tree")
    for name in selected:
        rel = Path(name)
        if rel.is_absolute() or ".." in rel.parts:
            raise RuntimeError(f"unsafe baseline path: {name}")
        target = destination / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(["jj", "--ignore-working-copy", "file", "show", "-r", revision, name], cwd=ROOT, check=True, stdout=subprocess.PIPE)
        target.write_bytes(result.stdout)
    (destination / "benchmarks").mkdir(exist_ok=True)
    (destination / "examples").mkdir(exist_ok=True)
    shutil.copy2(ROOT / "benchmarks/common.rs", destination / "benchmarks/common.rs")
    shutil.copy2(ROOT / "benchmarks/compare_v1.rs", destination / "examples/compare_v1.rs")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def quantile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def read_samples(path):
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["input_id"] = int(row["input_id"])
        for key in ("elapsed_ns", "prepare_ns", "call_ns", "output_copy_ns"):
            row[key] = int(row[key])
        row["output"] = [float(x) for x in row["output"].split(";")]
    return rows


def compare_outputs(first, second):
    if len(first) != len(second):
        raise RuntimeError("runners produced different sample counts")
    maximum = 0.0
    for a, b in zip(first, second):
        if a["input_id"] != b["input_id"] or len(a["output"]) != len(b["output"]):
            raise RuntimeError("runners produced different input order or output shape")
        for x, y in zip(a["output"], b["output"]):
            if not math.isfinite(x) or not math.isfinite(y):
                raise RuntimeError("non-finite output in benchmark CSV")
            difference = abs(x - y)
            maximum = max(maximum, difference)
            if difference > 1e-5 + 1e-3 * max(abs(x), abs(y)):
                raise RuntimeError(f"v1/v2 output mismatch: {x} vs {y} (input {a['input_id']})")
    return maximum


def stats(samples, field="elapsed_ns"):
    values = [row[field] / 1_000_000 for row in samples]
    return {"count": len(values), "min_ms": min(values), "median_ms": statistics.median(values),
            "p95_ms": quantile(values, 0.95), "p99_ms": quantile(values, 0.99), "max_ms": max(values)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="a0a081b0", help="JJ revision of the v1 implementation")
    parser.add_argument("--model", type=Path, default=ROOT / "demo/benchmarks/MobileNetV2Array.mlmodel")
    parser.add_argument("--original-model", type=Path, default=ROOT / "demo/models/MobileNetV2FP16.mlmodel")
    parser.add_argument("--labels", type=Path, help="labels generated with the array model")
    parser.add_argument("--case", choices=["relu", "mobilenet", "both"], default="both")
    parser.add_argument("--compute", choices=["cpu", "gpu", "ane"], default="gpu")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--pool", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261101)
    parser.add_argument("--swift-runtime-dir", type=Path, help="optional v1 Swift compatibility library directory")
    parser.add_argument("--output-dir", type=Path, help="new directory for raw CSVs and summary.json")
    args = parser.parse_args()
    if not (2 <= args.pool <= 1024 and args.iterations >= 2 * args.pool and 0 < args.seed <= 0xffffffff and args.warmup >= 0 and args.rounds > 0):
        parser.error("require pool 2..1024, iterations >= 2 * pool, nonnegative warmup, positive rounds and a nonzero u32 seed")
    cases = ["relu", "mobilenet"] if args.case == "both" else [args.case]
    args.model = args.model.resolve()
    args.original_model = args.original_model.resolve()
    args.labels = (args.labels or args.model.with_suffix(".labels.txt")).resolve()
    if "mobilenet" in cases:
        for path in (args.model, args.original_model, args.labels):
            if not path.is_file():
                parser.error(f"missing {path}; see docs/performance.md for model generation")
    now = datetime.now(timezone.utc)
    result_dir = (args.output_dir or ROOT / "demo/benchmarks/results" / now.strftime("%Y%m%dT%H%M%SZ")).resolve()
    result_dir.mkdir(parents=True, exist_ok=False)
    baseline_id = command(["jj", "--ignore-working-copy", "log", "--no-graph", "-r", args.baseline, "-T", "commit_id"], capture=True).strip()
    v2_id = command(["jj", "log", "--no-graph", "-r", "@", "-T", "commit_id"], capture=True).strip()
    metadata = {
        "date_utc": now.isoformat(), "baseline_revision": baseline_id, "v2_working_copy_revision": v2_id,
        "host": platform.machine(), "macos": platform.mac_ver()[0],
        "cpu": command(["sysctl", "-n", "machdep.cpu.brand_string"], capture=True).strip(),
        "rustc": command(["rustc", "--version"], capture=True).strip(),
        "sdk": command(["xcrun", "--show-sdk-version"], capture=True).strip(),
        "swift": command(["swift", "--version"], capture=True).strip(),
        "deployment_target": os.environ.get("MACOSX_DEPLOYMENT_TARGET"),
        "harness_sha256": {name: sha256(ROOT / name) for name in ("benchmarks/common.rs", "benchmarks/compare_v1.rs", "benchmarks/compare.py", "benchmarks/make_mobilenet_array.py", "examples/compare_v2.rs")},
        "relu_model_sha256": hashlib.sha256(RELU).hexdigest() if "relu" in cases else None,
        "case": args.case, "compute_units": args.compute, "iterations": args.iterations,
        "warmup": args.warmup, "pool": args.pool, "rounds": args.rounds, "seed": args.seed,
        "swift_runtime_dir": str(args.swift_runtime_dir) if args.swift_runtime_dir else None,
        "mobilenet_source": str(args.model) if "mobilenet" in cases else None,
        "mobilenet_sha256": sha256(args.model) if "mobilenet" in cases else None,
        "original_mobilenet_sha256": sha256(args.original_model) if "mobilenet" in cases else None,
        "labels_sha256": sha256(args.labels) if "mobilenet" in cases else None,
        "timing": "wall-clock per call; model compilation, loading, input generation and validation excluded",
    }
    (result_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    with tempfile.TemporaryDirectory(prefix="coreml-rs-compare-") as temp:
        tmp = Path(temp)
        baseline = tmp / "v1"
        baseline.mkdir()
        export_baseline(args.baseline, baseline)
        v2_build_env = os.environ.copy()
        v2_build_env["CARGO_TARGET_DIR"] = str(ROOT / "target")
        v1_build_env = os.environ.copy()
        v1_build_env["CARGO_TARGET_DIR"] = str(baseline / "target")
        command(["cargo", "build", "--locked", "--release", "--example", "compare_v2"], env=v2_build_env)
        command(["cargo", "build", "--locked", "--release", "--example", "compare_v1"], cwd=baseline, env=v1_build_env)
        v1 = baseline / "target/release/examples/compare_v1"
        v2 = ROOT / "target/release/examples/compare_v2"
        relu = tmp / "relu.mlmodel"
        relu.write_bytes(RELU)
        source = {"relu": relu, "mobilenet": args.model}
        compiled = {}
        for case in cases:
            compiled[case] = tmp / f"{case}.mlmodelc"
            command([str(v2), "compile", str(source[case]), str(compiled[case])])
        source_check = None
        if "mobilenet" in cases:
            original_compiled = tmp / "original-mobilenet.mlmodelc"
            command([str(v2), "compile", str(args.original_model), str(original_compiled)])
            source_check = command([str(v2), "validate-original", str(original_compiled), str(compiled["mobilenet"]), str(args.labels), args.compute], capture=True).strip()
            print(source_check, flush=True)
        order = random.Random(args.seed)
        summary = {"metadata": metadata, "source_model_check": source_check, "runs": [], "max_abs_output_difference": 0.0}
        v1_env = os.environ.copy()
        if args.swift_runtime_dir:
            path = str(args.swift_runtime_dir.resolve())
            v1_env["DYLD_LIBRARY_PATH"] = path + (":" + v1_env["DYLD_LIBRARY_PATH"] if v1_env.get("DYLD_LIBRARY_PATH") else "")
        for round_number in range(args.rounds):
            jobs = [(case, phase, version) for case in cases for phase in ("predict", "end-to-end") for version in ("v1", "v2")]
            jobs += [(case, "reuse", "v2") for case in cases]
            order.shuffle(jobs)
            rows = {}
            for case, phase, version in jobs:
                output = result_dir / f"round-{round_number + 1:02d}-{case}-{phase}-{version}.csv"
                executable = v1 if version == "v1" else v2
                command([str(executable), str(compiled[case]), case, phase, args.compute,
                         str(args.iterations), str(args.warmup), str(args.pool), str(args.seed), str(output)],
                        cwd=baseline if version == "v1" else ROOT, env=v1_env if version == "v1" else None)
                samples = read_samples(output)
                rows[(case, phase, version)] = samples
                entry = {"round": round_number + 1, "case": case, "phase": phase, "version": version,
                         "csv": output.name, "latency": stats(samples),
                         "stages": {field: stats(samples, field) for field in ("prepare_ns", "call_ns", "output_copy_ns")}}
                summary["runs"].append(entry)
                print(f"  {case:9} {phase:10} {version}: median {entry['latency']['median_ms']:.3f} ms, p95 {entry['latency']['p95_ms']:.3f} ms", flush=True)
            for case in cases:
                for phase in ("predict", "end-to-end"):
                    delta = compare_outputs(rows[(case, phase, "v1")], rows[(case, phase, "v2")])
                    summary["max_abs_output_difference"] = max(summary["max_abs_output_difference"], delta)
                delta = compare_outputs(rows[(case, "end-to-end", "v1")], rows[(case, "reuse", "v2")])
                summary["max_abs_output_difference"] = max(summary["max_abs_output_difference"], delta)
        summary["comparisons"] = []
        for case in cases:
            for phase in ("predict", "end-to-end"):
                def medians(version):
                    return [run["latency"]["median_ms"] for run in summary["runs"] if run["case"] == case and run["phase"] == phase and run["version"] == version]
                v1_median, v2_median = statistics.median(medians("v1")), statistics.median(medians("v2"))
                summary["comparisons"].append({"case": case, "phase": phase, "v1_median_of_round_medians_ms": v1_median,
                                               "v2_median_of_round_medians_ms": v2_median, "v2_relative_change_percent": (v2_median / v1_median - 1) * 100})
                print(f"{case} {phase}: v1 {v1_median:.3f} ms, v2 {v2_median:.3f} ms (median of round medians)", flush=True)
        (result_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Validated outputs. Results: {result_dir}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, subprocess.CalledProcessError, ValueError) as error:
        print(f"benchmark failed: {error}", file=sys.stderr)
        sys.exit(1)
