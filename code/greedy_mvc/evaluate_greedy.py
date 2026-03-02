#!/usr/bin/env python3
"""
Greedy MVC Baseline Evaluation

Implements greedy algorithms for Minimum Vertex Cover (MVC) and compares
against the S2V-DQN approach from "Learning Combinatorial Optimization
Algorithms over Graphs" (Dai et al., 2017).

Metrics: approximation ratio (solution / optimal), average VC size, time.
"""

import itertools
import os
import pickle
import sys
import time

import networkx as nx
import numpy as np

try:
    from scipy.optimize import milp, LinearConstraint, Bounds
    HAS_SCIPY_MILP = True
except ImportError:
    HAS_SCIPY_MILP = False

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ---------------------------------------------------------------------------
# Graph I/O
# ---------------------------------------------------------------------------

def load_graphs(pkl_path, num_graphs):
    """Load sequentially-pickled networkx graphs from a file."""
    graphs = []
    with open(pkl_path, 'rb') as f:
        for _ in range(num_graphs):
            graphs.append(pickle.load(f))
    return graphs


# ---------------------------------------------------------------------------
# Vertex cover verification
# ---------------------------------------------------------------------------

def verify_vertex_cover(G, cover_set):
    """Return True if cover_set is a valid vertex cover of G."""
    for u, v in G.edges():
        if u not in cover_set and v not in cover_set:
            return False
    return True


# ---------------------------------------------------------------------------
# Greedy algorithms
# ---------------------------------------------------------------------------

def greedy_static(G):
    """Static greedy: sort nodes by degree once, pick in order until covered.

    This matches the baseline in the paper (degree heuristic).
    Returns (vc_size, cover_set, elapsed_seconds).
    """
    t0 = time.time()
    covered_set = set()
    num_covered_edges = 0
    total_edges = G.number_of_edges()

    # Sort nodes by degree descending (computed once)
    nodes_by_degree = sorted(G.nodes(), key=lambda x: G.degree(x), reverse=True)

    pos = 0
    while num_covered_edges < total_edges:
        node = nodes_by_degree[pos]
        covered_set.add(node)
        for neigh in G.neighbors(node):
            if neigh not in covered_set:
                num_covered_edges += 1
        pos += 1

    elapsed = time.time() - t0
    return len(covered_set), covered_set, elapsed


def greedy_dynamic(G):
    """Dynamic greedy: at each step pick node with highest residual degree.

    Residual degree = number of uncovered incident edges.
    Returns (vc_size, cover_set, elapsed_seconds).
    """
    t0 = time.time()
    covered_set = set()
    # Track uncovered edges per node
    residual_degree = {v: G.degree(v) for v in G.nodes()}
    num_covered_edges = 0
    total_edges = G.number_of_edges()

    while num_covered_edges < total_edges:
        # Pick node with max residual degree
        best_node = max(
            (v for v in G.nodes() if v not in covered_set and residual_degree[v] > 0),
            key=lambda v: residual_degree[v]
        )
        covered_set.add(best_node)
        for neigh in G.neighbors(best_node):
            if neigh not in covered_set:
                num_covered_edges += 1
                residual_degree[neigh] -= 1
        residual_degree[best_node] = 0

    elapsed = time.time() - t0
    return len(covered_set), covered_set, elapsed


# ---------------------------------------------------------------------------
# Optimal MVC solvers
# ---------------------------------------------------------------------------

def optimal_mvc_bruteforce(G):
    """Exact MVC via brute-force enumeration. Feasible for n <= ~25."""
    nodes = list(G.nodes())
    n = len(nodes)
    edges = list(G.edges())

    for k in range(1, n + 1):
        for subset in itertools.combinations(nodes, k):
            cover = set(subset)
            if all(u in cover or v in cover for u, v in edges):
                return k, cover
    return n, set(nodes)


def optimal_mvc_ilp(G):
    """Exact MVC via Integer Linear Programming (scipy.optimize.milp)."""
    nodes = list(G.nodes())
    n = len(nodes)
    node_idx = {v: i for i, v in enumerate(nodes)}
    edges = list(G.edges())
    m = len(edges)

    # Minimize sum(x_i)
    c = np.ones(n)

    # Constraint: x_u + x_v >= 1 for each edge => -x_u - x_v <= -1
    A = np.zeros((m, n))
    for j, (u, v) in enumerate(edges):
        A[j, node_idx[u]] = -1.0
        A[j, node_idx[v]] = -1.0
    b_ub = -np.ones(m)

    constraints = LinearConstraint(A, ub=b_ub)
    integrality = np.ones(n)
    bounds = Bounds(lb=0, ub=1)

    result = milp(c, constraints=constraints, integrality=integrality, bounds=bounds)
    if result.success:
        selected = {nodes[i] for i in range(n) if result.x[i] > 0.5}
        return len(selected), selected
    else:
        # Fallback to brute force
        return optimal_mvc_bruteforce(G)


def optimal_mvc(G):
    """Compute exact MVC. Brute-force for n<=25, ILP for larger.

    Returns (vc_size, cover_set, elapsed_seconds).
    """
    t0 = time.time()
    n = G.number_of_nodes()
    if n <= 25:
        size, cover = optimal_mvc_bruteforce(G)
    elif HAS_SCIPY_MILP:
        size, cover = optimal_mvc_ilp(G)
    else:
        return None, None, 0.0
    elapsed = time.time() - t0
    return size, cover, elapsed


# ---------------------------------------------------------------------------
# S2V-DQN result parsing
# ---------------------------------------------------------------------------

def load_s2v_results(csv_path):
    """Parse S2V-DQN CSV results.

    Format per line: vc_size,count node0 node1 ...,time
    Returns list of dicts with keys: vc_size, nodes, time.
    """
    results = []
    with open(csv_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            vc_size = float(parts[0])
            node_str = parts[1].strip().split()
            count = int(node_str[0])
            nodes = [int(x) for x in node_str[1:count + 1]]
            elapsed = float(parts[2])
            results.append({'vc_size': int(vc_size), 'nodes': nodes, 'time': elapsed})
    return results


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_on_graphs(graphs, s2v_results=None, skip_optimal=False):
    """Run all methods on graphs, return list of per-graph result dicts."""
    results = []
    for i, G in enumerate(graphs):
        row = {
            'graph_idx': i,
            'num_nodes': G.number_of_nodes(),
            'num_edges': G.number_of_edges(),
        }

        # Greedy static
        sz, cover, t = greedy_static(G)
        assert verify_vertex_cover(G, cover), f'Static greedy invalid on graph {i}'
        row['static_size'] = sz
        row['static_time'] = t

        # Greedy dynamic
        sz, cover, t = greedy_dynamic(G)
        assert verify_vertex_cover(G, cover), f'Dynamic greedy invalid on graph {i}'
        row['dynamic_size'] = sz
        row['dynamic_time'] = t

        # Optimal
        if not skip_optimal:
            sz, cover, t = optimal_mvc(G)
            if sz is not None:
                assert verify_vertex_cover(G, cover), f'Optimal invalid on graph {i}'
                row['optimal_size'] = sz
                row['optimal_time'] = t
            else:
                row['optimal_size'] = None
                row['optimal_time'] = None
        else:
            row['optimal_size'] = None
            row['optimal_time'] = None

        # S2V-DQN
        if s2v_results is not None and i < len(s2v_results):
            row['s2v_size'] = s2v_results[i]['vc_size']
            row['s2v_time'] = s2v_results[i]['time']
        else:
            row['s2v_size'] = None
            row['s2v_time'] = None

        # Approximation ratios
        opt = row['optimal_size']
        if opt is not None and opt > 0:
            row['static_ratio'] = row['static_size'] / opt
            row['dynamic_ratio'] = row['dynamic_size'] / opt
            if row['s2v_size'] is not None:
                row['s2v_ratio'] = row['s2v_size'] / opt
            else:
                row['s2v_ratio'] = None
        else:
            row['static_ratio'] = None
            row['dynamic_ratio'] = None
            row['s2v_ratio'] = None

        results.append(row)
    return results


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def print_table(results, label):
    """Print a summary comparison table."""
    n = len(results)
    print(f'\n{"=" * 70}')
    print(f'  {label} ({n} graphs)')
    print(f'{"=" * 70}')

    methods = []

    # Optimal
    opt_sizes = [r['optimal_size'] for r in results if r['optimal_size'] is not None]
    opt_times = [r['optimal_time'] for r in results if r['optimal_time'] is not None]
    if opt_sizes:
        methods.append(('Optimal', opt_sizes, [1.0] * len(opt_sizes), opt_times))

    # Greedy static
    static_sizes = [r['static_size'] for r in results]
    static_times = [r['static_time'] for r in results]
    static_ratios = [r['static_ratio'] for r in results if r['static_ratio'] is not None]
    methods.append(('Greedy (static)', static_sizes, static_ratios, static_times))

    # Greedy dynamic
    dyn_sizes = [r['dynamic_size'] for r in results]
    dyn_times = [r['dynamic_time'] for r in results]
    dyn_ratios = [r['dynamic_ratio'] for r in results if r['dynamic_ratio'] is not None]
    methods.append(('Greedy (dynamic)', dyn_sizes, dyn_ratios, dyn_times))

    # S2V-DQN
    s2v_sizes = [r['s2v_size'] for r in results if r['s2v_size'] is not None]
    s2v_times = [r['s2v_time'] for r in results if r['s2v_time'] is not None]
    s2v_ratios = [r['s2v_ratio'] for r in results if r['s2v_ratio'] is not None]
    if s2v_sizes:
        methods.append(('S2V-DQN', s2v_sizes, s2v_ratios, s2v_times))

    header = f'{"Method":<20} | {"Avg VC Size":>11} | {"Avg Ratio":>9} | {"Worst Ratio":>11} | {"Avg Time (s)":>12}'
    print(header)
    print('-' * len(header))

    for name, sizes, ratios, times in methods:
        avg_size = np.mean(sizes) if sizes else float('nan')
        avg_ratio = np.mean(ratios) if ratios else float('nan')
        worst_ratio = np.max(ratios) if ratios else float('nan')
        avg_time = np.mean(times) if times else float('nan')
        print(f'{name:<20} | {avg_size:>11.2f} | {avg_ratio:>9.3f} | {worst_ratio:>11.3f} | {avg_time:>12.6f}')

    print()


def save_csv(results, output_path):
    """Save per-graph results to CSV."""
    keys = ['graph_idx', 'num_nodes', 'num_edges',
            'optimal_size', 'optimal_time',
            'static_size', 'static_time', 'static_ratio',
            'dynamic_size', 'dynamic_time', 'dynamic_ratio',
            's2v_size', 's2v_time', 's2v_ratio']
    with open(output_path, 'w') as f:
        f.write(','.join(keys) + '\n')
        for r in results:
            vals = []
            for k in keys:
                v = r.get(k)
                if v is None:
                    vals.append('')
                elif isinstance(v, float):
                    vals.append(f'{v:.6f}')
                else:
                    vals.append(str(v))
            f.write(','.join(vals) + '\n')
    print(f'Results saved to {output_path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    opt = {}
    for i in range(1, len(sys.argv), 2):
        opt[sys.argv[i][1:]] = sys.argv[i + 1]

    data_test = opt['data_test']
    num_graphs = int(opt.get('num_graphs', '100'))
    output_dir = opt.get('output_dir', 'results/greedy')
    s2v_results_path = opt.get('s2v_results', '')
    skip_optimal = opt.get('skip_optimal', '0') == '1'
    save_csv_flag = opt.get('save_csv', '0') == '1'

    os.makedirs(output_dir, exist_ok=True)

    print(f'Loading {num_graphs} graphs from {data_test}')
    graphs = load_graphs(data_test, num_graphs)

    s2v = None
    if s2v_results_path:
        print(f'Loading S2V-DQN results from {s2v_results_path}')
        s2v = load_s2v_results(s2v_results_path)

    results = evaluate_on_graphs(graphs, s2v_results=s2v, skip_optimal=skip_optimal)

    label = os.path.basename(data_test).replace('.pkl', '')
    print_table(results, label)

    if save_csv_flag:
        csv_path = os.path.join(output_dir, f'{label}-greedy.csv')
        save_csv(results, csv_path)
