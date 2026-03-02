#!/usr/bin/env python3
# greedy mvc evaluation on real-world memetracker data

import os
import sys
import time

import networkx as nx
import numpy as np

sys.path.append('%s/../memetracker' % os.path.dirname(os.path.realpath(__file__)))
from meme import build_full_graph


def verify_vertex_cover(G, cover_set):
    for u, v in G.edges():
        if u not in cover_set and v not in cover_set:
            return False
    return True


def greedy_static(G):
    """pick nodes by degree (sorted once) until all edges covered."""
    t0 = time.time()
    covered_set = set()
    num_covered_edges = 0
    total_edges = G.number_of_edges()

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
    """pick node with highest residual degree at each step."""
    t0 = time.time()
    covered_set = set()
    residual_degree = {v: G.degree(v) for v in G.nodes()}
    num_covered_edges = 0
    total_edges = G.number_of_edges()

    while num_covered_edges < total_edges:
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


if __name__ == '__main__':
    opt = {}
    for i in range(1, len(sys.argv), 2):
        opt[sys.argv[i][1:]] = sys.argv[i + 1]

    data_root = opt['data_root']
    output_dir = opt.get('output_dir', 'results/greedy')

    os.makedirs(output_dir, exist_ok=True)

    # load memetracker graph
    print('loading memetracker graph from %s' % data_root)
    g, _ = build_full_graph('%s/InfoNet5000Q1000NEXP.txt' % data_root, 'undirected')
    print('nodes: %d  edges: %d' % (g.number_of_nodes(), g.number_of_edges()))

    # static greedy
    sz, cover, t = greedy_static(g)
    assert verify_vertex_cover(g, cover)
    print('greedy static  - vc size: %d, time: %.4f' % (sz, t))

    # write csv in s2v format: vc_size,count node0 node1 ...,time
    result_file = '%s/greedy-static.csv' % output_dir
    with open(result_file, 'w') as f:
        f.write('%.8f,' % sz)
        nodes = sorted(cover)
        f.write('%d' % len(nodes))
        for n in nodes:
            f.write(' %d' % n)
        f.write(',%.6f\n' % t)
    print('results saved to %s' % result_file)

    # dynamic greedy
    sz, cover, t = greedy_dynamic(g)
    assert verify_vertex_cover(g, cover)
    print('greedy dynamic - vc size: %d, time: %.4f' % (sz, t))

    result_file = '%s/greedy-dynamic.csv' % output_dir
    with open(result_file, 'w') as f:
        f.write('%.8f,' % sz)
        nodes = sorted(cover)
        f.write('%d' % len(nodes))
        for n in nodes:
            f.write(' %d' % n)
        f.write(',%.6f\n' % t)
    print('results saved to %s' % result_file)
