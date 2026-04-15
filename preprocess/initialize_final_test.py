#!/usr/bin/env python3

from __future__ import annotations

import argparse
import concurrent.futures
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from preprocess.generate_final_test_cases import available_algorithms, load_artifacts, run_batch


FINAL_TEST_ROOT = REPO_ROOT / "datasets" / "final_test"
ENV_PATH = REPO_ROOT / ".env"
ENV_EXAMPLE_PATH = REPO_ROOT / ".env.example"
TARGET_LIMIT_ENV_NAME = "FINAL_TEST_TARGET_TIME_LIMIT"
CALIBRATION_REPORT_PATH = REPO_ROOT / "_data" / "final_test" / "_calibration_report.json"
RED = "\033[1;31m"
RESET = "\033[0m"

CANONICAL_SOLUTION_FILES = {
    "array-gray": "gray_solution.py",
    "array-moore": "moore_solution.py",
    "graph-mst-prim": "prim_solution.py",
    "graph-sp-bellmanford": "bf_sol.py",
    "graph-sp-dijkstra": "dijkstra_solution.py",
    "graph-sp-floyd": "floyd_solution.py",
    "math-euclidean": "euclidean.py",
    "math-strassen": "strassen_solution.py",
    "string-kmp": "kmp_sol.py",
    "string-manacher": "sol.py",
}


@dataclass(frozen=True)
class BenchmarkRecord:
    algo_name: str
    function_path: Path
    input_path: Path
    direct_time: float
    parallel_times: tuple[float, ...]
    average_parallel_time: float


@dataclass(frozen=True)
class AlgorithmBenchmarkSummary:
    algo_name: str
    slowest_case: BenchmarkRecord


def load_submit_execute():
    submit_tool_path = (
        REPO_ROOT
        / "simple_parallel"
        / "eval_logic"
        / "tool_module"
        / "tools"
        / "submit_final_answer.py"
    )
    spec = importlib.util.spec_from_file_location(
        "algo_test_submit_final_answer",
        submit_tool_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load module from {submit_tool_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.submit_execute


def log(message: str, quiet: bool) -> None:
    if not quiet:
        print(message)


def normalize_output_text(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.strip().splitlines())


def sanitize_solution_code(function_code: str) -> str:
    lines = function_code.splitlines()
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("if __name__ ==") and "__main__" in stripped:
            return "\n".join(lines[:index]) + "\n"
    return function_code


def ensure_env_file() -> None:
    if ENV_PATH.exists():
        return

    if ENV_EXAMPLE_PATH.exists():
        ENV_PATH.write_text(ENV_EXAMPLE_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    else:
        ENV_PATH.write_text("", encoding="utf-8")


def quote_env_value(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def upsert_env_value(env_name: str, env_value: str) -> None:
    ensure_env_file()
    target_line = f"{env_name}={quote_env_value(env_value)}"
    content = ENV_PATH.read_text(encoding="utf-8")
    lines = content.splitlines()

    replaced = False
    for index, line in enumerate(lines):
        if line.startswith(f"{env_name}="):
            lines[index] = target_line
            replaced = True
            break

    if not replaced:
        if lines and lines[-1] != "":
            lines.append("")
        lines.append(target_line)

    ENV_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def benchmark_case_once(
    submit_execute,
    algo_name: str,
    start_code: str,
    function_code: str,
    input_path: Path,
    output_path: Path,
    benchmark_time_limit: float,
) -> float:
    input_content = input_path.read_text(encoding="utf-8")
    stdout, stderr, message, target_func_time = submit_execute(
        code_parts=[start_code, function_code],
        input_content=input_content,
        time_limit=benchmark_time_limit,
    )
    if message != "Execution successful.":
        raise RuntimeError(
            f"benchmark failed for {algo_name} on {input_path.name}: {message}\n{stderr}"
        )

    expected_output = normalize_output_text(output_path.read_text(encoding="utf-8"))
    actual_output = normalize_output_text(stdout)
    if actual_output != expected_output:
        raise RuntimeError(
            f"benchmark output mismatch for {algo_name} on {input_path.name}"
        )

    if target_func_time < 0:
        raise RuntimeError(
            f"benchmark timing missing for {algo_name} on {input_path.name}"
        )

    return target_func_time


def write_calibration_report(
    threshold: float,
    margin: float,
    parallel_runs: int,
    algorithm_summaries: list[AlgorithmBenchmarkSummary],
    slowest_average_case: BenchmarkRecord,
) -> None:
    CALIBRATION_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report_payload = {
        "threshold": threshold,
        "margin": margin,
        "parallel_runs": parallel_runs,
        "slowest_average_case": {
            "algo_name": slowest_average_case.algo_name,
            "function_path": str(slowest_average_case.function_path),
            "input_path": str(slowest_average_case.input_path),
            "direct_time": slowest_average_case.direct_time,
            "parallel_times": list(slowest_average_case.parallel_times),
            "average_parallel_time": slowest_average_case.average_parallel_time,
            "difference_average_minus_direct": (
                slowest_average_case.average_parallel_time - slowest_average_case.direct_time
            ),
        },
        "algorithms": [
            {
                "algo_name": summary.algo_name,
                "slowest_case": {
                    "function_path": str(summary.slowest_case.function_path),
                    "input_path": str(summary.slowest_case.input_path),
                    "direct_time": summary.slowest_case.direct_time,
                    "parallel_times": list(summary.slowest_case.parallel_times),
                    "average_parallel_time": summary.slowest_case.average_parallel_time,
                    "difference_average_minus_direct": (
                        summary.slowest_case.average_parallel_time - summary.slowest_case.direct_time
                    ),
                },
            }
            for summary in algorithm_summaries
        ],
    }
    CALIBRATION_REPORT_PATH.write_text(
        json.dumps(report_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def generate_case_artifacts(algorithms: list[str], force: bool, jobs: int, quiet: bool) -> None:
    results = run_batch(algorithms, force=force, jobs=max(jobs, 1), verbose=not quiet)
    failures = [result for result in results if not result.success]
    if failures:
        details = "; ".join(f"{result.algo_name}: {result.message}" for result in failures)
        raise RuntimeError(f"final_test case generation failed: {details}")


def materialize_final_test_inputs(algorithms: list[str], copies: int, force: bool, quiet: bool) -> None:
    for algo_name in algorithms:
        algo_dir = FINAL_TEST_ROOT / algo_name
        for source_json_path in sorted(algo_dir.glob("level*/*.json")):
            level_name = source_json_path.parent.name
            prefix = f"id{source_json_path.stem}"
            output_dir = REPO_ROOT / "_data" / "final_test" / algo_name / level_name
            output_dir.mkdir(parents=True, exist_ok=True)

            expected_paths = [output_dir / f"{prefix}_{index}.json" for index in range(copies)]
            if not force and all(path.exists() for path in expected_paths):
                continue

            payload = json.loads(source_json_path.read_text(encoding="utf-8"))
            final_item = {
                "question": payload["problem"],
                "test_cases": payload["test_cases"],
            }
            rendered = json.dumps(final_item, indent=2, ensure_ascii=False) + "\n"
            for output_path in expected_paths:
                output_path.write_text(rendered, encoding="utf-8")

        log(f"[materialize] {algo_name}", quiet)


def benchmark_algorithms(
    algorithms: list[str],
    benchmark_time_limit: float,
    margin: float,
    parallel_runs: int,
    quiet: bool,
) -> tuple[float, list[AlgorithmBenchmarkSummary], BenchmarkRecord]:
    submit_execute = load_submit_execute()
    algorithm_summaries: list[AlgorithmBenchmarkSummary] = []

    for algo_name in algorithms:
        generator_dir = FINAL_TEST_ROOT / algo_name / "_generator"
        start_code_path = generator_dir / "start_code.py"
        function_path = generator_dir / CANONICAL_SOLUTION_FILES[algo_name]

        start_code = start_code_path.read_text(encoding="utf-8")
        function_code = sanitize_solution_code(function_path.read_text(encoding="utf-8"))

        case_records: list[BenchmarkRecord] = []
        for artifact in load_artifacts(algo_name):
            direct_time = benchmark_case_once(
                submit_execute=submit_execute,
                algo_name=algo_name,
                start_code=start_code,
                function_code=function_code,
                input_path=artifact.input_path,
                output_path=artifact.output_path,
                benchmark_time_limit=benchmark_time_limit,
            )
            with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_runs) as executor:
                futures = [
                    executor.submit(
                        benchmark_case_once,
                        submit_execute,
                        algo_name,
                        start_code,
                        function_code,
                        artifact.input_path,
                        artifact.output_path,
                        benchmark_time_limit,
                    )
                    for _ in range(parallel_runs)
                ]
                parallel_times = tuple(future.result() for future in futures)

            average_parallel_time = sum(parallel_times) / len(parallel_times)
            case_records.append(
                BenchmarkRecord(
                    algo_name=algo_name,
                    function_path=function_path,
                    input_path=artifact.input_path,
                    direct_time=direct_time,
                    parallel_times=parallel_times,
                    average_parallel_time=average_parallel_time,
                )
            )

        slowest_case = max(case_records, key=lambda item: item.average_parallel_time)
        algorithm_summaries.append(
            AlgorithmBenchmarkSummary(
                algo_name=algo_name,
                slowest_case=slowest_case,
            )
        )
        log(
            (
                f"[benchmark] {algo_name}: avg_max={slowest_case.average_parallel_time:.6f}s "
                f"direct={slowest_case.direct_time:.6f}s "
                f"delta={slowest_case.average_parallel_time - slowest_case.direct_time:+.6f}s"
            ),
            quiet,
        )

    slowest_average_case = max(
        (summary.slowest_case for summary in algorithm_summaries),
        key=lambda item: item.average_parallel_time,
    )
    threshold = slowest_average_case.average_parallel_time + margin
    write_calibration_report(
        threshold=threshold,
        margin=margin,
        parallel_runs=parallel_runs,
        algorithm_summaries=algorithm_summaries,
        slowest_average_case=slowest_average_case,
    )
    return threshold, algorithm_summaries, slowest_average_case


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Initialize final_test data and calibrate a machine-specific runtime threshold.",
    )
    parser.add_argument("--jobs", type=int, default=4, help="Parallel jobs for test-case generation.")
    parser.add_argument("--copies", type=int, default=16, help="Number of duplicated final_test prompts per source JSON.")
    parser.add_argument("--margin", type=float, default=0.1, help="Safety margin added to the slowest canonical runtime.")
    parser.add_argument(
        "--benchmark-time-limit",
        type=float,
        default=30.0,
        help="Timeout in seconds used while benchmarking canonical solutions.",
    )
    parser.add_argument(
        "--benchmark-parallel-runs",
        type=int,
        default=3,
        help="Number of concurrent benchmark runs used to average each final_test case.",
    )
    parser.add_argument("--force", action="store_true", help="Regenerate files even if they already exist.")
    parser.add_argument("--quiet", action="store_true", help="Reduce logging.")
    args = parser.parse_args(argv)

    algorithms = available_algorithms()
    missing_solutions = sorted(set(algorithms) - set(CANONICAL_SOLUTION_FILES))
    if missing_solutions:
        raise RuntimeError(
            f"missing canonical solution mapping for: {', '.join(missing_solutions)}"
        )

    upsert_env_value("PROJECT_ROOT", str(REPO_ROOT))

    log("[initialize] generating final_test case artifacts", args.quiet)
    generate_case_artifacts(algorithms, force=args.force, jobs=args.jobs, quiet=args.quiet)

    log("[initialize] materializing _data/final_test", args.quiet)
    materialize_final_test_inputs(algorithms, copies=args.copies, force=args.force, quiet=args.quiet)

    log("[initialize] benchmarking canonical solutions", args.quiet)
    threshold, algorithm_summaries, slowest_average_case = benchmark_algorithms(
        algorithms,
        benchmark_time_limit=args.benchmark_time_limit,
        margin=args.margin,
        parallel_runs=max(args.benchmark_parallel_runs, 1),
        quiet=args.quiet,
    )

    upsert_env_value(TARGET_LIMIT_ENV_NAME, f"{threshold:.6f}")

    if not args.quiet:
        avg_direct_diff = (
            slowest_average_case.average_parallel_time - slowest_average_case.direct_time
        )
        print("")
        print("Initialization complete")
        print(f"- slowest case   : {slowest_average_case.algo_name} / {slowest_average_case.input_path.name}")
        print(f"- slowest avg    : {slowest_average_case.average_parallel_time:.6f}s")
        print(f"- slowest direct : {slowest_average_case.direct_time:.6f}s")
        print(f"- avg-direct diff: {avg_direct_diff:+.6f}s")
        print(f"- margin         : {args.margin:.6f}s")
        print(f"- target limit   : {threshold:.6f}s")
        print(f"- report         : {CALIBRATION_REPORT_PATH}")
        print(f"- written to     : {ENV_PATH}")
        if avg_direct_diff > 0.1:
            print(
                f"{RED}WARNING: current machine shows a large throughput gap under concurrent load. "
                "The calibrated Reinvention time threshold may differ noticeably from single-run speed, "
                "so time-based pass rates should not be treated as strictly stable on this machine."
                f"{RESET}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
