#!/usr/bin/env python3
"""Aggregate MVC experiment outputs into analysis-ready tables.

This script:
1) Merges synthetic greedy CSVs with S2V eval CSVs (latest per dataset id).
2) Writes per-graph and aggregated synthetic summaries.
3) Writes a small real-world summary table.
4) Prints headline metrics to stdout.
"""

from __future__ import annotations

import csv
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

SYN_GREEDY_GLOB = "code/greedy_mvc/results/greedy/*-greedy.csv"
SYN_S2V_GLOB = "code/s2v_mvc/results/**/test-*.csv"
REALWORLD_S2V = Path(
    "code/realworld_s2v_mvc/results/dqn-meme/embed-64-nbp-1-rh-64-prob_q-7/gnn-5-300.csv"
)
REALWORLD_GREEDY_STATIC = Path("code/realworld_greedy_mvc/results/greedy/greedy-static.csv")
REALWORLD_GREEDY_DYNAMIC = Path("code/realworld_greedy_mvc/results/greedy/greedy-dynamic.csv")


def as_float(value: str | None) -> float | None:
    if value is None:
        return None
    s = value.strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def mean(values: list[float | None]) -> float | None:
    valid = [v for v in values if v is not None]
    if not valid:
        return None
    return sum(valid) / len(valid)


def fmt(v: float | None) -> str:
    return "" if v is None else f"{v:.6f}"


def parse_dataset_meta(dataset_id: str) -> dict[str, str]:
    # Example:
    # gtype-barabasi_albert-nrange-100-200-n_graph-1000-p-0.00-m-4-w-float-0-1-cnctd-0-seed-2
    meta: dict[str, str] = {"dataset_id": dataset_id}
    gtype = re.search(r"gtype-([^-]+(?:_[^-]+)*)", dataset_id)
    nrange = re.search(r"nrange-(\d+-\d+)", dataset_id)
    n_graph = re.search(r"n_graph-(\d+)", dataset_id)
    p = re.search(r"-p-([^-]+)", dataset_id)
    seed = re.search(r"-seed-(\d+)", dataset_id)
    if gtype:
        meta["gtype"] = gtype.group(1)
    if nrange:
        meta["nrange"] = nrange.group(1)
    if n_graph:
        meta["n_graph"] = n_graph.group(1)
    if p:
        meta["p"] = p.group(1)
    if seed:
        meta["seed"] = seed.group(1)
    return meta


def dataset_id_from_s2v_file(path: Path) -> str | None:
    name = path.name
    if not name.startswith("test-"):
        return None
    rest = name[len("test-") :]
    if ".pkl-gnn-train-" in rest:
        return rest.split(".pkl-gnn-train-", 1)[0]
    if ".pkl-gnn-" in rest:
        return rest.split(".pkl-gnn-", 1)[0]
    return None


def parse_s2v_rows(path: Path) -> list[tuple[int, float, float]]:
    rows: list[tuple[int, float, float]] = []
    with path.open("r", newline="") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 3:
                continue
            size = as_float(parts[0])
            t = as_float(parts[-1])
            if size is None or t is None:
                continue
            rows.append((idx, size, t))
    return rows


def parse_cover_line(path: Path) -> tuple[float, float]:
    # Format: "<size>,<k> <nodes...>,<time>"
    with path.open("r", newline="") as f:
        line = f.readline().strip()
    parts = line.split(",")
    if len(parts) < 3:
        raise ValueError(f"unexpected cover line format in {path}")
    size = float(parts[0])
    t = float(parts[-1])
    return size, t


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    # pick latest S2V result per dataset id
    s2v_latest: dict[str, dict[str, object]] = {}
    for path in REPO_ROOT.glob(SYN_S2V_GLOB):
        dataset_id = dataset_id_from_s2v_file(path)
        if dataset_id is None:
            continue
        mtime = path.stat().st_mtime
        existing = s2v_latest.get(dataset_id)
        if existing is None or mtime > float(existing["mtime"]):
            s2v_latest[dataset_id] = {
                "mtime": mtime,
                "path": path,
                "rows": parse_s2v_rows(path),
            }

    # per-graph synthetic merge table
    synthetic_rows: list[dict[str, str]] = []
    for gpath in sorted(REPO_ROOT.glob(SYN_GREEDY_GLOB)):
        dataset_id = gpath.name[: -len("-greedy.csv")]
        meta = parse_dataset_meta(dataset_id)
        s2v_entry = s2v_latest.get(dataset_id)
        s2v_rows = list(s2v_entry["rows"]) if s2v_entry is not None else []
        s2v_source = (
            str(Path(s2v_entry["path"]).relative_to(REPO_ROOT)) if s2v_entry is not None else ""
        )

        with gpath.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                graph_idx = int(row["graph_idx"])
                optimal_size = as_float(row.get("optimal_size"))
                static_ratio = as_float(row.get("static_ratio"))
                dynamic_ratio = as_float(row.get("dynamic_ratio"))
                static_time = as_float(row.get("static_time"))
                dynamic_time = as_float(row.get("dynamic_time"))
                s2v_size = None
                s2v_time = None
                s2v_ratio = None
                if graph_idx < len(s2v_rows):
                    _, s2v_size, s2v_time = s2v_rows[graph_idx]
                    if optimal_size and optimal_size > 0:
                        s2v_ratio = s2v_size / optimal_size

                out = {
                    "dataset_id": dataset_id,
                    "gtype": meta.get("gtype", ""),
                    "nrange": meta.get("nrange", ""),
                    "n_graph": meta.get("n_graph", ""),
                    "p": meta.get("p", ""),
                    "seed": meta.get("seed", ""),
                    "graph_idx": str(graph_idx),
                    "num_nodes": row.get("num_nodes", ""),
                    "num_edges": row.get("num_edges", ""),
                    "optimal_size": row.get("optimal_size", ""),
                    "static_ratio": fmt(static_ratio),
                    "dynamic_ratio": fmt(dynamic_ratio),
                    "s2v_ratio": fmt(s2v_ratio),
                    "static_time": fmt(static_time),
                    "dynamic_time": fmt(dynamic_time),
                    "s2v_time": fmt(s2v_time),
                    "greedy_source": str(gpath.relative_to(REPO_ROOT)),
                    "s2v_source": s2v_source,
                }
                synthetic_rows.append(out)

    synthetic_rows.sort(
        key=lambda r: (
            r["gtype"],
            tuple(int(x) for x in r["nrange"].split("-")) if r["nrange"] else (math.inf, math.inf),
            int(r["graph_idx"]),
        )
    )

    out_dir = REPO_ROOT / "results" / "analysis"
    write_csv(
        out_dir / "summary_synthetic_per_graph.csv",
        [
            "dataset_id",
            "gtype",
            "nrange",
            "n_graph",
            "p",
            "seed",
            "graph_idx",
            "num_nodes",
            "num_edges",
            "optimal_size",
            "static_ratio",
            "dynamic_ratio",
            "s2v_ratio",
            "static_time",
            "dynamic_time",
            "s2v_time",
            "greedy_source",
            "s2v_source",
        ],
        synthetic_rows,
    )

    # aggregate by dataset
    by_dataset: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for r in synthetic_rows:
        by_dataset[(r["dataset_id"], r["gtype"], r["nrange"])].append(r)

    dataset_rows: list[dict[str, str]] = []
    for (dataset_id, gtype, nrange), rows in sorted(by_dataset.items()):
        static_ratios = [as_float(r["static_ratio"]) for r in rows]
        dynamic_ratios = [as_float(r["dynamic_ratio"]) for r in rows]
        s2v_ratios = [as_float(r["s2v_ratio"]) for r in rows]
        static_times = [as_float(r["static_time"]) for r in rows]
        dynamic_times = [as_float(r["dynamic_time"]) for r in rows]
        s2v_times = [as_float(r["s2v_time"]) for r in rows]

        # per-row winner on ratio (lower is better)
        wins = Counter()
        for r in rows:
            candidates = {
                "static": as_float(r["static_ratio"]),
                "dynamic": as_float(r["dynamic_ratio"]),
                "s2v": as_float(r["s2v_ratio"]),
            }
            candidates = {k: v for k, v in candidates.items() if v is not None}
            if candidates:
                winner = min(candidates.items(), key=lambda kv: kv[1])[0]
                wins[winner] += 1

        dataset_rows.append(
            {
                "dataset_id": dataset_id,
                "gtype": gtype,
                "nrange": nrange,
                "num_graphs": str(len(rows)),
                "static_ratio_mean": fmt(mean(static_ratios)),
                "dynamic_ratio_mean": fmt(mean(dynamic_ratios)),
                "s2v_ratio_mean": fmt(mean(s2v_ratios)),
                "static_time_mean": fmt(mean(static_times)),
                "dynamic_time_mean": fmt(mean(dynamic_times)),
                "s2v_time_mean": fmt(mean(s2v_times)),
                "winner_static_count": str(wins["static"]),
                "winner_dynamic_count": str(wins["dynamic"]),
                "winner_s2v_count": str(wins["s2v"]),
            }
        )

    write_csv(
        out_dir / "summary_synthetic_bucket.csv",
        [
            "dataset_id",
            "gtype",
            "nrange",
            "num_graphs",
            "static_ratio_mean",
            "dynamic_ratio_mean",
            "s2v_ratio_mean",
            "static_time_mean",
            "dynamic_time_mean",
            "s2v_time_mean",
            "winner_static_count",
            "winner_dynamic_count",
            "winner_s2v_count",
        ],
        dataset_rows,
    )

    # aggregate by graph family
    by_family: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in synthetic_rows:
        by_family[r["gtype"]].append(r)

    family_rows: list[dict[str, str]] = []
    for gtype, rows in sorted(by_family.items()):
        family_rows.append(
            {
                "gtype": gtype,
                "num_rows": str(len(rows)),
                "static_ratio_mean": fmt(mean([as_float(r["static_ratio"]) for r in rows])),
                "dynamic_ratio_mean": fmt(mean([as_float(r["dynamic_ratio"]) for r in rows])),
                "s2v_ratio_mean": fmt(mean([as_float(r["s2v_ratio"]) for r in rows])),
                "static_time_mean": fmt(mean([as_float(r["static_time"]) for r in rows])),
                "dynamic_time_mean": fmt(mean([as_float(r["dynamic_time"]) for r in rows])),
                "s2v_time_mean": fmt(mean([as_float(r["s2v_time"]) for r in rows])),
            }
        )

    write_csv(
        out_dir / "summary_synthetic_family.csv",
        [
            "gtype",
            "num_rows",
            "static_ratio_mean",
            "dynamic_ratio_mean",
            "s2v_ratio_mean",
            "static_time_mean",
            "dynamic_time_mean",
            "s2v_time_mean",
        ],
        family_rows,
    )

    # real-world summary
    realworld_rows: list[dict[str, str]] = []
    s2v_size, s2v_time = parse_cover_line(REPO_ROOT / REALWORLD_S2V)
    static_size, static_time = parse_cover_line(REPO_ROOT / REALWORLD_GREEDY_STATIC)
    dynamic_size, dynamic_time = parse_cover_line(REPO_ROOT / REALWORLD_GREEDY_DYNAMIC)
    best_size = min(s2v_size, static_size, dynamic_size)
    realworld_rows.extend(
        [
            {
                "method": "s2v",
                "vc_size": f"{s2v_size:.6f}",
                "time_s": f"{s2v_time:.6f}",
                "ratio_vs_best": f"{s2v_size / best_size:.6f}",
            },
            {
                "method": "greedy_static",
                "vc_size": f"{static_size:.6f}",
                "time_s": f"{static_time:.6f}",
                "ratio_vs_best": f"{static_size / best_size:.6f}",
            },
            {
                "method": "greedy_dynamic",
                "vc_size": f"{dynamic_size:.6f}",
                "time_s": f"{dynamic_time:.6f}",
                "ratio_vs_best": f"{dynamic_size / best_size:.6f}",
            },
        ]
    )
    write_csv(out_dir / "summary_realworld.csv", ["method", "vc_size", "time_s", "ratio_vs_best"], realworld_rows)

    # headline console stats
    static_all = [as_float(r["static_ratio"]) for r in synthetic_rows]
    dynamic_all = [as_float(r["dynamic_ratio"]) for r in synthetic_rows]
    s2v_all = [as_float(r["s2v_ratio"]) for r in synthetic_rows]
    print("Wrote:")
    print(f"  {out_dir / 'summary_synthetic_per_graph.csv'}")
    print(f"  {out_dir / 'summary_synthetic_bucket.csv'}")
    print(f"  {out_dir / 'summary_synthetic_family.csv'}")
    print(f"  {out_dir / 'summary_realworld.csv'}")
    print()
    print(f"Synthetic rows: {len(synthetic_rows)}")
    print(f"Synthetic mean ratio static : {fmt(mean(static_all))}")
    print(f"Synthetic mean ratio dynamic: {fmt(mean(dynamic_all))}")
    print(f"Synthetic mean ratio s2v    : {fmt(mean(s2v_all))}")
    print()
    print(f"Real-world S2V vc size: {s2v_size:.1f} (time {s2v_time:.4f}s)")
    print(f"Real-world static size: {static_size:.1f} (time {static_time:.4f}s)")
    print(f"Real-world dynamic size: {dynamic_size:.1f} (time {dynamic_time:.4f}s)")


if __name__ == "__main__":
    main()
