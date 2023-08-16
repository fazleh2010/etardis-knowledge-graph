#!/usr/bin/env python3

import csv
import os
import pickle

from src.crawling.dbpedia_handler import DBPediaHandler
from src.crawling.graph_handler2 import GraphHandler
from neo4j import GraphDatabase


def extract_categories(do_cleanup: bool, update_index: bool, depth: int,data_path:str, node_name:str, connetion_name:str,updated_labels_path:str,g:GraphHandler):
    #url = "bolt://localhost:7474"

    print("EXTRACT pickled data")
    nodes = g.extract_pickle(node_name)
    connections = g.extract_pickle(connetion_name)

    if do_cleanup:
        print("DELETE current graph data")
        g.delete_graph()

    updated_labels_lookup = dict()

    if update_index:
        print("CREATE index structure containing available node labels")
        g.create_index(nodes,updated_labels_path)

    return

def get_categories(updated_labels_path:str):

    line_number=1
    updated_labels_lookup=dict()
    with open(updated_labels_path, "r") as infile:
        reader = csv.reader(infile, delimiter=",")
        for row in reader:
            uri = row[0]
            label = str(row[1])
            updated_labels_lookup[uri] = label
            #print(str(line_number)+" uri::"+uri+" label::"+label)
            line_number=line_number+1

    return updated_labels_lookup
def insert_into_neo4j(do_cleanup: bool, update_index: bool, depth: int,data_path:str, node_name:str, connetion_name:str,g:GraphHandler,updated_labels_lookup:dict):
    """
    Additional step to insert crawled and pickled data into graph database.
    :param do_cleanup: specify if database should be deleted first
    :param update_index: specify if new labels are set and indices (uri) have to be set
    :param depth: specify the search depth starting from entry point uri (depth is equal to number of connections in a specific direction)
    :return:
    to run this script we need to run neo4j data first
    """
    #//url = "bolt://localhost:7474"
    #url = "bolt://localhost:7687"
    #username = "neo4j"
    #password = "password"
    data_path_label = "/media/elahi/Elements/A-Projects/etradis/resources/data/"
    nodes_path = os.path.join(data_path, node_name)
    connections_path = os.path.join(data_path, connetion_name)
    #updated_labels_path = os.path.join(data_path, "updated_labels_test.csv")

    print("EXTRACT pickled data")
    nodes = g.extract_pickle(nodes_path)
    connections = g.extract_pickle(connections_path)

    if do_cleanup:
        print("DELETE current graph data")
        g.delete_graph()

    print(node_name+" "+connections_path)
    print("INSERT nodes")
    g.insert_nodes(nodes, updated_labels_lookup)

    print("INSERT connections")
    g.insert_connections(connections, updated_labels_lookup)


def main():
    """
    Initialize crawling and insertion process for a given dbpedia entry uri.
    :return:
    """
    do_cleanup = False
    update_index = True
    depth = 1
    data_path = "/media/elahi/Elements/A-Projects/etradis/resources/data/"

    url = "bolt://localhost:7687"
    username = "neo4j"
    password = "password"
    #graph = GraphHandler(url, username, password)
    updated_labels_path = os.path.join(data_path, "updated_labels.csv")

    etradis_category = [ "http://localhost:9999/etradis/TopicalConcept"]

    updated_labels_lookup = dict()
    for uri in etradis_category:
       batch = uri.replace("http://localhost:9999/etradis/", "")
       for file in os.listdir(data_path):
           nodeMat = "nodes_d_" + batch + "_"
           konMat = "connections_d_" + batch + "_"
           if file.startswith(nodeMat):
               nodes_path = os.path.join(data_path, file)
               strFile = file.replace(data_path, "")
               strFile = file.replace(nodeMat, konMat)
               connections_path = os.path.join(data_path, strFile)
               updated_labels_path = os.path.join(data_path, "updated_labels_test.csv")
               #updated_labels_lookup=extract_categories(do_cleanup, update_index, depth, data_path, nodes_path, connections_path, updated_labels_path,graph)
               #updated_labels_lookup=get_categories(updated_labels_path)
               print(nodes_path+" "+connections_path+" "+str(len(updated_labels_lookup)))
               #insert_into_neo4j(do_cleanup, update_index, depth, data_path, nodes_path, connections_path,graph,updated_labels_lookup)



if __name__ == "__main__":
    main()