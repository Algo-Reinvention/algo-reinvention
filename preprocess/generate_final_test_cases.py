'''
cd $PROJECT_ROOT
python preprocess/generate_final_test_cases.py --jobs 4
'''

from __future__ import annotations

import argparse
import concurrent.futures
import heapq
import json
import subprocess
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = REPO_ROOT / "datasets/final_test"


@dataclass(frozen=True)
class CaseArtifact:
    input_path: Path
    output_path: Path


@dataclass(frozen=True)
class AlgorithmHandler:
    generate_inputs: Callable[["GenerationContext"], None]
    generate_outputs: Callable[["GenerationContext"], None] | None = None


@dataclass
class GenerationContext:
    algo_name: str
    algo_dir: Path
    generator_dir: Path
    case_dir: Path
    artifacts: tuple[CaseArtifact, ...]
    force: bool = False
    verbose: bool = True

    def log(self, message: str) -> None:
        if self.verbose:
            print(f"[{self.algo_name}] {message}")


@dataclass(frozen=True)
class RunResult:
    algo_name: str
    success: bool
    message: str


def available_algorithms() -> list[str]:
    return sorted(
        path.name
        for path in DATASET_ROOT.iterdir()
        if path.is_dir() and (path / "_generator").is_dir()
    )


def load_artifacts(algo_name: str) -> tuple[CaseArtifact, ...]:
    algo_dir = DATASET_ROOT / algo_name
    artifacts: dict[tuple[str, str], CaseArtifact] = {}
    for json_path in sorted(algo_dir.glob("level*/*.json")):
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        for case in payload.get("test_cases", []):
            input_path = REPO_ROOT / case["input_path"]
            output_path = REPO_ROOT / case["output_path"]
            key = (str(input_path), str(output_path))
            artifacts[key] = CaseArtifact(input_path=input_path, output_path=output_path)
    return tuple(sorted(artifacts.values(), key=lambda item: str(item.input_path)))


def build_context(algo_name: str, force: bool, verbose: bool) -> GenerationContext:
    algo_dir = DATASET_ROOT / algo_name
    generator_dir = algo_dir / "_generator"
    return GenerationContext(
        algo_name=algo_name,
        algo_dir=algo_dir,
        generator_dir=generator_dir,
        case_dir=generator_dir / "test_cases",
        artifacts=load_artifacts(algo_name),
        force=force,
        verbose=verbose,
    )


def missing_inputs(ctx: GenerationContext) -> list[Path]:
    return [artifact.input_path for artifact in ctx.artifacts if not artifact.input_path.exists()]


def missing_outputs(ctx: GenerationContext) -> list[Path]:
    return [artifact.output_path for artifact in ctx.artifacts if not artifact.output_path.exists()]


def all_required_files_exist(ctx: GenerationContext) -> bool:
    return all(
        artifact.input_path.exists() and artifact.output_path.exists()
        for artifact in ctx.artifacts
    )


def ensure_case_dir(ctx: GenerationContext) -> None:
    ctx.case_dir.mkdir(parents=True, exist_ok=True)


def run_python_script(ctx: GenerationContext, script_name: str, *args: str) -> None:
    script_path = ctx.generator_dir / script_name
    command = [sys.executable, str(script_path), *args]
    ctx.log(f"running {' '.join(command)}")
    result = subprocess.run(
        command,
        cwd=ctx.generator_dir,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed: {' '.join(command)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    if result.stdout.strip():
        ctx.log(result.stdout.strip())
    if result.stderr.strip():
        ctx.log(result.stderr.strip())


def write_output_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if content.endswith("\n"):
        path.write_text(content, encoding="utf-8")
    else:
        path.write_text(content + "\n", encoding="utf-8")


def generate_inputs_default_gen(ctx: GenerationContext) -> None:
    if not ctx.force and not missing_inputs(ctx):
        ctx.log("all input files already exist; skipping gen.py")
        return
    run_python_script(ctx, "gen.py")


def generate_inputs_graph_mst_prim(ctx: GenerationContext) -> None:
    target = ctx.case_dir / "n2600.txt"
    if not ctx.force and target.exists():
        ctx.log("all input files already exist; skipping graph-mst-prim generation")
        return
    run_python_script(ctx, "gen.py", "--n", "2600", "--out", "test_cases/n2600.txt")


def generate_inputs_graph_sp_bellmanford(ctx: GenerationContext) -> None:
    if ctx.force or not (ctx.case_dir / "n500.txt").exists():
        run_python_script(ctx, "gen.py", "--node", "500", "--output", "test_cases/n500.txt")
    else:
        ctx.log("n500.txt already exists; skipping gen.py")

    if ctx.force or not (ctx.case_dir / "hack_dij1.txt").exists():
        run_python_script(ctx, "hack_dij1.py")
    else:
        ctx.log("hack_dij1.txt already exists; skipping hack_dij1.py")

    if ctx.force or not (ctx.case_dir / "hack_dij2.txt").exists():
        run_python_script(
            ctx,
            "hack_dij2.py",
            "--node",
            "350",
            "--output",
            "test_cases/hack_dij2.txt",
        )
    else:
        ctx.log("hack_dij2.txt already exists; skipping hack_dij2.py")

    if ctx.force or not (ctx.case_dir / "hack_dij3.txt").exists():
        run_python_script(
            ctx,
            "hack_dij3.py",
            "--node",
            "350",
            "--output",
            "test_cases/hack_dij3.txt",
        )
    else:
        ctx.log("hack_dij3.txt already exists; skipping hack_dij3.py")

    if ctx.force or not (ctx.case_dir / "hack_dij4.txt").exists():
        run_python_script(ctx, "hack_dij4.py")
    else:
        ctx.log("hack_dij4.txt already exists; skipping hack_dij4.py")


def generate_inputs_graph_sp_dijkstra(ctx: GenerationContext) -> None:
    heap_killer_path = ctx.case_dir / "dijkstra_heap_killer.txt"
    spfa_killer_path = ctx.case_dir / "spfa_killer.txt"

    if ctx.force or not heap_killer_path.exists():
        ctx.log(f"writing {heap_killer_path.name}")
        write_dijkstra_heap_killer(heap_killer_path, n=2600, source=1)
    else:
        ctx.log(f"{heap_killer_path.name} already exists; skipping")

    if ctx.force or not spfa_killer_path.exists():
        ctx.log(f"writing {spfa_killer_path.name}")
        write_spfa_dense_hack_graph(spfa_killer_path, n=2600)
    else:
        ctx.log(f"{spfa_killer_path.name} already exists; skipping")


def generate_inputs_graph_sp_floyd(ctx: GenerationContext) -> None:
    if ctx.force or not (ctx.case_dir / "n300_dense.txt").exists():
        run_python_script(ctx, "gen.py", "150", "22350", "test_cases/n300_dense.txt")
    else:
        ctx.log("n300_dense.txt already exists; skipping gen.py")

    if ctx.force or not (ctx.case_dir / "spfa_killer.txt").exists():
        run_python_script(ctx, "spfa_killer_gen.py", "150", "test_cases/spfa_killer.txt")
    else:
        ctx.log("spfa_killer.txt already exists; skipping spfa_killer_gen.py")


def write_dijkstra_heap_killer(path: Path, n: int, source: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"{n}\n")
        handle.write(f"{n * (n - 1)}\n")
        for k in range(1, n + 1):
            for j in range(1, n + 1):
                if k == j:
                    continue
                weight = 1 if j == k + 1 else 2 * n - 2 * k
                handle.write(f"{k} {j} {weight}\n")
        handle.write(f"{source}\n")


def write_spfa_dense_hack_graph(path: Path, n: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    base_val = 1_000_000_000
    noise_weight = n * base_val
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"{n}\n")
        handle.write(f"{n * (n - 1)}\n")
        for u in range(1, n + 1):
            for v in range(1, n + 1):
                if u == v:
                    continue
                if u == 1:
                    weight = base_val // v
                elif v == u - 1 and v != 1:
                    weight = 1
                else:
                    weight = noise_weight
                handle.write(f"{u} {v} {weight}\n")
        handle.write("1\n")


def parse_directed_graph_with_source(path: Path) -> tuple[int, list[list[tuple[int, int]]], int, bool]:
    with path.open("r", encoding="utf-8") as handle:
        n = int(handle.readline().strip())
        m = int(handle.readline().strip())
        graph: list[list[tuple[int, int]]] = [[] for _ in range(n + 1)]
        forward_only = True
        for _ in range(m):
            u, v, w = map(int, handle.readline().split())
            if u >= v:
                forward_only = False
            graph[u].append((v, w))
        source = int(handle.readline().strip())
    return n, graph, source, forward_only


def parse_undirected_graph(path: Path) -> tuple[int, list[list[tuple[int, int]]]]:
    with path.open("r", encoding="utf-8") as handle:
        n = int(handle.readline().strip())
        m = int(handle.readline().strip())
        graph: list[list[tuple[int, int]]] = [[] for _ in range(n + 1)]
        for _ in range(m):
            u, v, w = map(int, handle.readline().split())
            graph[u].append((v, w))
            graph[v].append((u, w))
    return n, graph


def parse_floyd_input(path: Path) -> tuple[int, list[list[float]]]:
    inf = float("inf")
    with path.open("r", encoding="utf-8") as handle:
        n = int(handle.readline().strip())
        m = int(handle.readline().strip())
        dist = [[inf] * (n + 1) for _ in range(n + 1)]
        for i in range(1, n + 1):
            dist[i][i] = 0
        for _ in range(m):
            u, v, w = map(int, handle.readline().split())
            if w < dist[u][v]:
                dist[u][v] = w
    return n, dist


def format_distances(dist: list[int], inf: int) -> str:
    return " ".join("-1" if value >= inf else str(value) for value in dist[1:])


def solve_dijkstra(n: int, graph: list[list[tuple[int, int]]], source: int) -> list[int]:
    inf = 10**30
    dist = [inf] * (n + 1)
    dist[source] = 0
    heap: list[tuple[int, int]] = [(0, source)]
    while heap:
        current_dist, u = heapq.heappop(heap)
        if current_dist != dist[u]:
            continue
        for v, w in graph[u]:
            new_dist = current_dist + w
            if new_dist < dist[v]:
                dist[v] = new_dist
                heapq.heappush(heap, (new_dist, v))
    return dist


def solve_forward_dag_shortest_paths(
    n: int,
    graph: list[list[tuple[int, int]]],
    source: int,
) -> list[int]:
    inf = 10**30
    dist = [inf] * (n + 1)
    dist[source] = 0
    for u in range(1, n + 1):
        if dist[u] >= inf:
            continue
        base = dist[u]
        for v, w in graph[u]:
            new_dist = base + w
            if new_dist < dist[v]:
                dist[v] = new_dist
    return dist


def solve_spfa_or_detect_negative_cycle(
    n: int,
    graph: list[list[tuple[int, int]]],
    source: int,
) -> tuple[bool, list[int]]:
    inf = 10**30
    dist = [inf] * (n + 1)
    dist[source] = 0
    path_len = [0] * (n + 1)
    in_queue = [False] * (n + 1)
    queue: deque[int] = deque([source])
    in_queue[source] = True

    while queue:
        u = queue.popleft()
        in_queue[u] = False
        base = dist[u]
        for v, w in graph[u]:
            new_dist = base + w
            if new_dist < dist[v]:
                dist[v] = new_dist
                path_len[v] = path_len[u] + 1
                if path_len[v] >= n:
                    return True, dist
                if not in_queue[v]:
                    queue.append(v)
                    in_queue[v] = True

    return False, dist


def solve_prim_mst(n: int, graph: list[list[tuple[int, int]]]) -> int | None:
    visited = [False] * (n + 1)
    heap: list[tuple[int, int]] = [(0, 1)]
    total_weight = 0
    count = 0

    while heap:
        weight, u = heapq.heappop(heap)
        if visited[u]:
            continue
        visited[u] = True
        total_weight += weight
        count += 1
        for v, edge_weight in graph[u]:
            if not visited[v]:
                heapq.heappush(heap, (edge_weight, v))

    return total_weight if count == n else None


def solve_floyd(n: int, dist: list[list[float]]) -> list[str]:
    inf = float("inf")
    for k in range(1, n + 1):
        dist_k = dist[k]
        for i in range(1, n + 1):
            dik = dist[i][k]
            if dik == inf:
                continue
            row_i = dist[i]
            for j in range(1, n + 1):
                dkj = dist_k[j]
                if dkj == inf:
                    continue
                new_dist = dik + dkj
                if new_dist < row_i[j]:
                    row_i[j] = new_dist

    lines = []
    for i in range(1, n + 1):
        row = []
        for j in range(1, n + 1):
            value = dist[i][j]
            row.append("INF" if value == inf else str(int(value)))
        lines.append(" ".join(row))
    return lines


def generate_outputs_array_moore(ctx: GenerationContext) -> None:
    # The bundled generator constructs every case so that 1000000001 is the majority element.
    for artifact in ctx.artifacts:
        if not ctx.force and artifact.output_path.exists():
            ctx.log(f"{artifact.output_path.name} already exists; skipping")
            continue
        write_output_text(artifact.output_path, "1000000001")
        ctx.log(f"wrote {artifact.output_path.name}")


def generate_outputs_graph_mst_prim(ctx: GenerationContext) -> None:
    for artifact in ctx.artifacts:
        if not ctx.force and artifact.output_path.exists():
            ctx.log(f"{artifact.output_path.name} already exists; skipping")
            continue
        n, graph = parse_undirected_graph(artifact.input_path)
        total_weight = solve_prim_mst(n, graph)
        content = str(total_weight) if total_weight is not None else "impossible"
        write_output_text(artifact.output_path, content)
        ctx.log(f"wrote {artifact.output_path.name}")


def generate_outputs_graph_sp_bellmanford(ctx: GenerationContext) -> None:
    inf = 10**30
    for artifact in ctx.artifacts:
        if not ctx.force and artifact.output_path.exists():
            ctx.log(f"{artifact.output_path.name} already exists; skipping")
            continue
        n, graph, source, forward_only = parse_directed_graph_with_source(artifact.input_path)
        if forward_only:
            dist = solve_forward_dag_shortest_paths(n, graph, source)
            content = format_distances(dist, inf)
        else:
            has_negative_cycle, dist = solve_spfa_or_detect_negative_cycle(n, graph, source)
            content = "Negative Cycle" if has_negative_cycle else format_distances(dist, inf)
        write_output_text(artifact.output_path, content)
        ctx.log(f"wrote {artifact.output_path.name}")


def generate_outputs_graph_sp_dijkstra(ctx: GenerationContext) -> None:
    inf = 10**30
    for artifact in ctx.artifacts:
        if not ctx.force and artifact.output_path.exists():
            ctx.log(f"{artifact.output_path.name} already exists; skipping")
            continue
        n, graph, source, _ = parse_directed_graph_with_source(artifact.input_path)
        dist = solve_dijkstra(n, graph, source)
        write_output_text(artifact.output_path, format_distances(dist, inf))
        ctx.log(f"wrote {artifact.output_path.name}")


def generate_outputs_graph_sp_floyd(ctx: GenerationContext) -> None:
    for artifact in ctx.artifacts:
        if not ctx.force and artifact.output_path.exists():
            ctx.log(f"{artifact.output_path.name} already exists; skipping")
            continue
        n, dist = parse_floyd_input(artifact.input_path)
        lines = solve_floyd(n, dist)
        write_output_text(artifact.output_path, "\n".join(lines))
        ctx.log(f"wrote {artifact.output_path.name}")


HANDLERS: dict[str, AlgorithmHandler] = {
    "array-gray": AlgorithmHandler(generate_inputs=generate_inputs_default_gen),
    "array-moore": AlgorithmHandler(
        generate_inputs=generate_inputs_default_gen,
        generate_outputs=generate_outputs_array_moore,
    ),
    "graph-mst-prim": AlgorithmHandler(
        generate_inputs=generate_inputs_graph_mst_prim,
        generate_outputs=generate_outputs_graph_mst_prim,
    ),
    "graph-sp-bellmanford": AlgorithmHandler(
        generate_inputs=generate_inputs_graph_sp_bellmanford,
        generate_outputs=generate_outputs_graph_sp_bellmanford,
    ),
    "graph-sp-dijkstra": AlgorithmHandler(
        generate_inputs=generate_inputs_graph_sp_dijkstra,
        generate_outputs=generate_outputs_graph_sp_dijkstra,
    ),
    "graph-sp-floyd": AlgorithmHandler(
        generate_inputs=generate_inputs_graph_sp_floyd,
        generate_outputs=generate_outputs_graph_sp_floyd,
    ),
    "math-euclidean": AlgorithmHandler(generate_inputs=generate_inputs_default_gen),
    "math-strassen": AlgorithmHandler(generate_inputs=generate_inputs_default_gen),
    "string-kmp": AlgorithmHandler(generate_inputs=generate_inputs_default_gen),
    "string-manacher": AlgorithmHandler(generate_inputs=generate_inputs_default_gen),
}


def run_algorithm(algo_name: str, force: bool = False, verbose: bool = True) -> RunResult:
    ctx = build_context(algo_name, force=force, verbose=verbose)
    handler = HANDLERS[algo_name]
    try:
        ensure_case_dir(ctx)
        if not ctx.force and all_required_files_exist(ctx):
            ctx.log("all required files already exist; nothing to do")
            return RunResult(algo_name=algo_name, success=True, message="already complete")

        handler.generate_inputs(ctx)

        still_missing_inputs = missing_inputs(ctx)
        if still_missing_inputs:
            missing_names = ", ".join(path.name for path in still_missing_inputs)
            raise RuntimeError(f"missing input files after generation: {missing_names}")

        if handler.generate_outputs is not None:
            handler.generate_outputs(ctx)

        still_missing_outputs = missing_outputs(ctx)
        if still_missing_outputs:
            missing_names = ", ".join(path.name for path in still_missing_outputs)
            raise RuntimeError(f"missing output files after generation: {missing_names}")

        return RunResult(algo_name=algo_name, success=True, message="ok")
    except Exception as exc:  # pragma: no cover - surfaced to CLI
        return RunResult(algo_name=algo_name, success=False, message=str(exc))


def normalize_algorithms(selected: list[str] | None) -> list[str]:
    available = available_algorithms()
    if not selected:
        return available

    invalid = sorted(set(selected) - set(available))
    if invalid:
        raise ValueError(f"unknown algorithms: {', '.join(invalid)}")
    return selected


def run_batch(algorithms: list[str], force: bool, jobs: int, verbose: bool) -> list[RunResult]:
    if jobs <= 1 or len(algorithms) <= 1:
        return [run_algorithm(algo_name, force=force, verbose=verbose) for algo_name in algorithms]

    results: list[RunResult] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as executor:
        future_map = {
            executor.submit(run_algorithm, algo_name, force, verbose): algo_name
            for algo_name in algorithms
        }
        for future in concurrent.futures.as_completed(future_map):
            results.append(future.result())
    results.sort(key=lambda item: item.algo_name)
    return results


def run_single_algorithm_cli(generator_dir: Path, argv: list[str] | None = None) -> int:
    algo_name = generator_dir.parent.name
    parser = argparse.ArgumentParser(
        description=f"Generate final-test inputs and outputs for {algo_name}.",
    )
    parser.add_argument("--force", action="store_true", help="Regenerate files even if they already exist.")
    parser.add_argument("--quiet", action="store_true", help="Reduce logging.")
    args = parser.parse_args(argv)

    result = run_algorithm(algo_name, force=args.force, verbose=not args.quiet)
    if result.success:
        if not args.quiet:
            print(f"[{algo_name}] done")
        return 0

    print(f"[{algo_name}] failed: {result.message}", file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate datasets/final_test cases through unified per-algorithm entrypoints.",
    )
    parser.add_argument(
        "--algo",
        action="append",
        help="Algorithm name to generate. Repeat the flag to select multiple algorithms. Default: all.",
    )
    parser.add_argument("--force", action="store_true", help="Regenerate files even if they already exist.")
    parser.add_argument("--jobs", type=int, default=1, help="Number of algorithms to run in parallel.")
    parser.add_argument("--quiet", action="store_true", help="Reduce logging.")
    parser.add_argument("--list", action="store_true", help="List available algorithms and exit.")
    args = parser.parse_args(argv)

    if args.list:
        for algo_name in available_algorithms():
            print(algo_name)
        return 0

    algorithms = normalize_algorithms(args.algo)
    results = run_batch(algorithms, force=args.force, jobs=max(args.jobs, 1), verbose=not args.quiet)

    failures = [result for result in results if not result.success]
    if failures:
        for result in failures:
            print(f"[{result.algo_name}] failed: {result.message}", file=sys.stderr)
        return 1

    if not args.quiet:
        print(f"completed {len(results)} algorithm(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
