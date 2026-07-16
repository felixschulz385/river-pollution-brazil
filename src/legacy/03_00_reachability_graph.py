import os
from functools import lru_cache, partial
import multiprocessing as mp
from tqdm import tqdm
import pandas as pd
import geopandas as gpd
import networkx as nx
import shelve


def create_graph(edge_list):
    """ Create a directed graph from a list of edges. """
    G = nx.DiGraph()
    G.add_edges_from(edge_list)
    return G

@lru_cache(maxsize=None)
def find_reachable_nodes(G, start_node):
    """ Find all reachable nodes from a given node using DFS. """
    # We use a set to avoid duplicates
    visited = set()
    # Stack for DFS
    stack = [start_node]
    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            # Add all neighbors to the stack
            stack.extend(G.successors(node))
    return frozenset(visited)  # Return as an immutable set for caching

def compute_reachability_parallel(graph):
    """ Compute reachability for each node in the graph in parallel. """
    
    # If the graph is small, we don't need multiprocessing
    if len(graph) < 1000:
        n_cores = 1
    else:
        n_cores = int(os.environ["SLURM_CPUS_PER_TASK"])
    
    with mp.Pool(n_cores) as pool:
        # Get all nodes in the graph
        nodes = list(graph.nodes())
        # Prepare the partial function with fixed graph
        func = partial(find_reachable_nodes, graph)
        # Map the function across all nodes using multiprocessing
        results = pool.map(func, nodes)
        # Turn frozensets back into sets
        results = [set(x) for x in results]
        # Return the results as a dictionary
        return zip(nodes, results)

def compute_reachability():
    # Import the river network shapefile
    rivers_brazil_shapefile = gpd.read_feather("/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/data/river_network/shapefile.feather")
    
    # Iterate over all estuaries
    for i in tqdm(rivers_brazil_shapefile.estuary.dropna().unique()):
        edges = rivers_brazil_shapefile.query(f"estuary=={i}").copy().loc[:, ["downstream_node_id", "upstream_node_id"]].dropna().values
        # Create the graph
        G = create_graph(edges)
        # Compute the reachability
        with shelve.open("/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/data/river_network/reachability.db") as reachability:
            for x in compute_reachability_parallel(G):
                reachability[str(int(x[0]))] = x[1] 