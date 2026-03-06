import numpy as np
import networkx as nx
import pickle as cp
import random
import ctypes
import os
import sys
import time
from tqdm import tqdm

sys.path.append( '%s/mvc_lib' % os.path.dirname(os.path.realpath(__file__)) )
from mvc_lib import MvcLib

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


def resolve_path(path, must_exist=False):
    if os.path.isabs(path):
        candidate = path
    else:
        script_relative = os.path.join(SCRIPT_DIR, path)
        candidate = path if os.path.exists(path) else script_relative

    if must_exist and not os.path.exists(candidate):
        raise FileNotFoundError(
            "Could not find path %r. Checked %r and %r." % (
                path,
                os.path.abspath(path),
                os.path.abspath(os.path.join(SCRIPT_DIR, path)),
            )
        )
    return candidate


def normalize_graph(g):
    legacy_adj = getattr(g, 'adj', None)
    legacy_node = getattr(g, 'node', None)
    if isinstance(legacy_adj, dict) and isinstance(legacy_node, dict):
        rebuilt = nx.Graph()
        rebuilt.graph.update(getattr(g, 'graph', {}))
        for node, attrs in legacy_node.items():
            rebuilt.add_node(node, **attrs)
        for u, nbrs in legacy_adj.items():
            for v, edge_attrs in nbrs.items():
                if rebuilt.has_edge(u, v):
                    continue
                rebuilt.add_edge(u, v, **edge_attrs)
        return rebuilt
    return g
    
def find_model_file(opt):
    max_n = int(opt.get('model_max_n', opt['max_n']))
    min_n = int(opt.get('model_min_n', opt['min_n']))
    save_dir = resolve_path(opt['save_dir'], must_exist=True)
    log_file = '%s/log-%d-%d.txt' % (save_dir, min_n, max_n)
    if not os.path.isfile(log_file):
        return None

    best_r = 1000000
    best_it = -1
    with open(log_file, 'r') as f:
        for line in f:
            if 'average' in line:
                line = line.split(' ')
                it = int(line[1].strip())
                r = float(line[-1].strip())
                if r < best_r:
                    best_r = r
                    best_it = it
    if best_it < 0:
        return None
    print('using iter=', best_it, 'with r=', best_r)
    return '%s/nrange_%d_%d_iter_%d.model' % (save_dir, min_n, max_n, best_it)
    
if __name__ == '__main__':
    api = MvcLib(sys.argv)
    
    opt = {}
    for i in range(1, len(sys.argv), 2):
        opt[sys.argv[i][1:]] = sys.argv[i + 1]

    model_file = find_model_file(opt)
    if model_file is None:
        print(
            'no trained model found in %s for nrange %s-%s; skipping'
            % (resolve_path(opt['save_dir'], must_exist=True), opt['min_n'], opt['max_n'])
        )
        sys.exit(0)
    print('loading', model_file)
    sys.stdout.flush()
    api.LoadModel(model_file)

    n_test = 1000
    data_test = resolve_path(opt['data_test'], must_exist=True)
    f = open(data_test, 'rb')
    frac = 0.0

    save_dir = resolve_path(opt['save_dir'], must_exist=True)
    test_name = os.path.basename(data_test)
    model_min_n = opt.get('model_min_n', opt['min_n'])
    model_max_n = opt.get('model_max_n', opt['max_n'])
    if model_min_n == opt['min_n'] and model_max_n == opt['max_n']:
        result_file = '%s/test-%s-gnn-%s-%s.csv' % (save_dir, test_name, opt['min_n'], opt['max_n'])
    else:
        result_file = '%s/test-%s-gnn-train-%s-%s-test-%s-%s.csv' % (
            save_dir,
            test_name,
            model_min_n,
            model_max_n,
            opt['min_n'],
            opt['max_n'],
        )

    with open(result_file, 'w') as f_out:
        print('testing')
        sys.stdout.flush()
        for i in tqdm(range(n_test)):
            g = normalize_graph(cp.load(f))
            api.InsertGraph(g, is_test=True)
            t1 = time.time()
            val, sol = api.GetSol(i, nx.number_of_nodes(g))
            t2 = time.time()
            f_out.write('%.8f,' % val)
            f_out.write('%d' % sol[0])
            for i in range(sol[0]):
                f_out.write(' %d' % sol[i + 1])
            f_out.write(',%.6f\n' % (t2 - t1))
            frac += val
    print('average size of vc: ', frac / n_test)
