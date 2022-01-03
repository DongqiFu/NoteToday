import time
import math
import numpy as np
from collections import defaultdict
from numpy.linalg import inv
import networkx as nx


# --- parameters ---
damping = 0.5  # for lazy random walk
motif_order = 2  # an edge is a 2nd-order motif
c1 = 200
c4 = 140
b = 3
output_file = open('stats.txt', 'a')


def build(graph_name):
    graph = open(graph_name, 'r')
    edges = graph.readlines()
    graph.close()

    g = nx.Graph()

    for edge in edges[1:]:
        items = edge.strip().split(',')
        node_0 = items[0]
        node_1 = items[1]
        g.add_node(node_0)
        g.add_node(node_1)
        g.add_edge(node_0, node_1, weight=1.0, transition_probability=1.0)

    return g


def power_method(graph, pagerank, damping=0.5):
    pagerank_last = pagerank
    pagerank = dict.fromkeys(pagerank_last.keys(), 0)
    for node in pagerank:
        for neighbor in graph[node]:
            pagerank[node] += damping * pagerank_last[neighbor] * graph[node][neighbor]['transition_probability']
        pagerank[node] += (1.0 - damping) * pagerank_last[node]
    return pagerank


def adjacency_to_transition_graph(graph):
    digraph = graph.to_directed()
    for node in graph:
        for neighbor in graph[node]:
            digraph[node][neighbor]['transition_probability'] = digraph[node][neighbor]['weight'] / graph.degree(neighbor, weight='weight')
    return digraph


def nibble(graph, seed, upper_bound_phi, damping=0.5):
    # hyperparameters
    num_nodes = graph.number_of_nodes()
    mu_v = sum(
        dict(graph.in_degree(weight='weight')).values()  # change to in degree for directed graphs?
    )
    l = math.ceil(math.log(mu_v, 2) / 2)
    t_last = (l + 1) * math.ceil(
        (2 / math.pow(upper_bound_phi, 2)) * math.log(c1 * (l + 2) * math.sqrt(mu_v / 2), math.e)
    )
    t_max = int(math.ceil(t_last))
    epsilon = 1 / (1800 * (l + 2) * t_last * math.pow(2, b))

    # initialization
    pagerank = dict.fromkeys(graph, 0.0)
    pagerank[seed] = 1.0

    # cold start: 3 iterations of power method
    for itr in range(3):
        start_time = time.time()
        pagerank = power_method(graph, pagerank, damping=damping)
        pagerank_sum = sum(pagerank.values())
        for node in pagerank:
            pagerank[node] /= pagerank_sum
        end_time = time.time()
        with open('stats.txt', 'a') as output_file:
            output_file.write('[Cold Start] Elapsed Time: {time} seconds\n'.format(time=end_time-start_time))


    # -------Sweep Cut----------
    start_time = time.time()
    for t_iter in range(t_max):
        pagerank = power_method(graph, pagerank, damping=damping)
        pagerank_sum = sum(pagerank.values())
        for node in pagerank:
            pagerank[node] /= pagerank_sum

        score = {node: pagerank[node] / graph.in_degree(node, weight='weight') for node in graph}
        score_ranking = sorted(score, key=score.get, reverse=True)

        # need to modify this part for high-order motifs
        next_cluster = score_ranking[:motif_order]
        next_volume = sum(
            dict(graph.in_degree(next_cluster, weight='weight')).values()
        )
        next_temp = graph.subgraph(next_cluster).size(weight='weight')
        next_phi = (next_volume - next_temp) / min(next_volume, mu_v - next_volume)

        for j in range(motif_order, num_nodes):
            cluster = next_cluster
            volume = next_volume
            phi = next_phi

            next_cluster = score_ranking[:j+1]
            next_volume = sum(
                dict(graph.in_degree(next_cluster, weight='weight')).values()
            )
            next_temp = graph.subgraph(next_cluster).size()
            next_phi = (next_volume - next_temp) / min(next_volume, mu_v - next_volume)

            indexing_node = cluster[-1]
            I_x = pagerank[indexing_node] / graph.in_degree(indexing_node, weight='weight')

            condition_small_phi = phi <= upper_bound_phi
            condition_phi_decresing = phi < next_phi
            condition_volume = math.pow(2, b) <= volume <= (5 / 6) * mu_v
            condition_escape = I_x >= epsilon / (c4 * (l + 2) * math.pow(2, b))

            if condition_escape and condition_volume and condition_small_phi and condition_phi_decresing:
                end_time = time.time()
                with open('stats.txt', 'a') as output_file:
                    output_file.write('[Sweep Cut] Cluster found!\tElapsed Time: {time} seconds\n'.format(time=end_time-start_time))
                return cluster, phi

    # no cluster found
    end_time = time.time()
    with open('stats.txt', 'a') as output_file:
        output_file.write('[Sweep Cut] No cluster found!\tElapsed Time: {time} seconds\n'.format(time=end_time-start_time))
    return [], None


if __name__ == '__main__':
    filepath_list = ['data/alpha/alpha_t10']

    # each graph
    for filepath in filepath_list:
        print('-----------------------------------------------------')
        print("Data set: " + filepath.split('/')[1])

        # build graph
        graph = filepath + '.txt'
        start_time = time.time()
        g = build(graph)
        print('nnodes', g.number_of_nodes(), 'nedges', g.number_of_edges())
        g = adjacency_to_transition_graph(g)
        build_time = time.time() - start_time
        print('Nibble build time:' + str(build_time))

        # -----------------------------------------
        seed_node_list = ['0', '500']
        for seed_node in seed_node_list:
            print('---> seed_node: ' + str(seed_node))
            upper_bound_phi = 0.5
            start_time = time.time()
            cluster, phi = nibble(g, seed_node, upper_bound_phi)
            clustering_time = time.time() - start_time
            if len(cluster) != 0:
                print('Nibble conductance: ' + str(phi))
                print('Nibble clustering time: ' + str(clustering_time))
                np.save('data_nibble/' + filepath.split('/')[1] + '_seed_node_' + str(seed_node) + '_clus_vec', cluster)
            else:
                print('---> seed_node: ' + str(seed_node) + 'does not find local cluster under conductance ' + str(
                    upper_bound_phi))
