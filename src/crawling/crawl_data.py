#!/usr/bin/env python3

import csv
import os
import pickle

from src.crawling.dbpedia_handler import DBPediaHandler
from src.crawling.graph_handler2 import GraphHandler


def crawl_dbpedia(start_uri: str, depth: int, node_appearance_threshold: int):
    """
    For a given entry URI crawl data from dbpedia which consists of entry node and adjacent nodes of a given depth.
    Crawled nodes and corresponding relations are stored as pickled files.
    :param start_uri: entry uri
    :param depth: search depth starting from entry node
    :param node_appearance_threshold: remove nodes with fewer connections
    :return:
    """
    data_path = "/home/elahi/A-etardis/data/"
    search_depth = depth
    min_node_appearance = node_appearance_threshold
    d = DBPediaHandler()

    print("CRAWL data from DBPedia sparql endpoint")
    d.retrieve_data(start_uri=start_uri, search_depth=search_depth,
                    filter_threshold=min_node_appearance)

    print("DUMP files for nodes and connections")
    with open(os.path.join(data_path, f"nodes_d{search_depth}.pkl"), "wb") as nodes_file:
        pickle.dump(d.nodes, nodes_file)

    with open(os.path.join(data_path, f"connections_d{search_depth}.pkl"), "wb") as connections_file:
        pickle.dump(d.connections, connections_file)

    # metrics
    # depth 1
    # number nodes before filtering: 2425
    # number connections before filtering:  1278496
    # number nodes after filtering: 2371
    # number connections after filtering:  501700 (final filtering 42823)

    # depth 2
    # number nodes before filtering: 46880
    # number connections before filtering:  26446815
    # number nodes after filtering: 46880
    # number connections after filtering:  3363171


def insert_into_neo4j(do_cleanup: bool, update_index: bool, depth: int):
    """
    Additional step to insert crawled and pickled data into graph database.
    :param do_cleanup: specify if database should be deleted first
    :param update_index: specify if new labels are set and indices (uri) have to be set
    :param depth: specify the search depth starting from entry point uri (depth is equal to number of connections in a specific direction)
    :return:
    """
    url = "bolt://localhost:7687"
    username = "neo4j"
    password = "password"
    data_path = "/home/elahi/A-etardis/data/"
    nodes_path = os.path.join(data_path, f"nodes_d{depth}.pkl")
    connections_path = os.path.join(data_path, f"connections_d{depth}.pkl")
    updated_labels_path = os.path.join(data_path, "updated_labels.csv")

    g = GraphHandler(url, username, password)

    print("EXTRACT pickled data")
    nodes = g.extract_pickle(nodes_path)
    connections = g.extract_pickle(connections_path)

    if do_cleanup:
        print("DELETE current graph data")
        g.delete_graph()

    if update_index:
        print("CREATE index structure containing available node labels")
        g.create_index(nodes, updated_labels_path)

    updated_labels_lookup = dict()
    with open(updated_labels_path, "r") as infile:
        reader = csv.reader(infile, delimiter=",")
        for row in reader:
            uri = row[0]
            # labels = ast.literal_eval(row[1])
            label = str(row[1])
            updated_labels_lookup[uri] = label

    print("INSERT nodes")
    g.insert_nodes(nodes, updated_labels_lookup)

    print("INSERT connections")
    g.insert_connections(connections, updated_labels_lookup)


def main():
    """
    Initialize crawling and insertion process for a given dbpedia entry uri.
    :return:
    """
    search_depth = 1
    # uri = "http://dbpedia.org/resource/Hundred_Years'_War"
    uri ="http://dbpedia.org/resource/Albert_Einstein"
    #crawl_dbpedia(start_uri=uri, depth=search_depth, node_appearance_threshold=3)
    insert_into_neo4j(do_cleanup=False, update_index=True, depth=search_depth)


if __name__ == "__main__":
    main()
