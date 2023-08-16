#!/usr/bin/env python3

import csv
import os
import pickle

from src.crawling.dbpedia_handler import DBPediaHandler
from src.crawling.graph_handler import GraphHandler
from neo4j import GraphDatabase


def crawl_dbpedia(start_uri: str, depth: int, node_appearance_threshold: int):
    """
    For a given entry URI crawl data from dbpedia which consists of entry node and adjacent nodes of a given depth.
    Crawled nodes and corresponding relations are stored as pickled files.
    :param start_uri: entry uri
    :param depth: search depth starting from entry node
    :param node_appearance_threshold: remove nodes with fewer connections
    :return:
    """
    search_depth = depth
    min_node_appearance = node_appearance_threshold
    d = DBPediaHandler()

    print("CRAWL data from DBPedia sparql endpoint")
    d.retrieve_data(start_uri=start_uri, search_depth=search_depth,
                    filter_threshold=min_node_appearance)
    return d

def get_categories(updated_labels_path:str):

    line_number=1
    done_uris=dict()
    with open(updated_labels_path, "r") as infile:
        reader = csv.reader(infile, delimiter=",")
        for row in reader:
            uri = str(row[0])
            done_uris[uri] = str(row[1])
            #print(str(line_number)+" uri::"+uri+" label::"+label)
            line_number=line_number+1

    return done_uris

def insert_into_neo4j(do_cleanup: bool, update_index: bool, depth: int,data_path:str, node_name:str, connetion_name:str,g:GraphHandler):
    """
    Additional step to insert crawled and pickled data into graph database.
    :param do_cleanup: specify if database should be deleted first
    :param update_index: specify if new labels are set and indices (uri) have to be set
    :param depth: specify the search depth starting from entry point uri (depth is equal to number of connections in a specific direction)
    :return:
    to run this script we need to run neo4j data first
    """

    nodes_path = os.path.join(data_path, node_name)
    connections_path = os.path.join(data_path, connetion_name)

    print("EXTRACT pickled data")
    nodes = g.extract_pickle(nodes_path)
    connections = g.extract_pickle(connections_path)

    if do_cleanup:
        print("DELETE current graph data")
        g.delete_graph()

    print(node_name+" "+connections_path)
    print("INSERT nodes")
    g.insert_nodes(nodes)

    print("INSERT connections")
    g.insert_connections(connections)

def main(uri_set=None):
    """
    Initialize crawling and insertion process for a given dbpedia entry uri.
    :return:
    """
    do_cleanup = False
    update_index = True
    depth = 1
    search_depth = 1
    data_path = "/media/elahi/Elements/A-Projects/etradis/resources/dataT/"
    done_uris_path=data_path+"updated_done_uris_event.csv"
    done_uris=get_categories(done_uris_path)

    d = DBPediaHandler()
    etradis_category = ["http://localhost:9999/etradis/Event"]

    url = "bolt://localhost:7687"
    username = "neo4j"
    password = "password"
    graph = GraphHandler(url, username, password)

    # take too long
    # total_resource_class = d.get_count_basuris("http://localhost:9999/etradis/MaterialObject")
    # print(total_resource_class)
    total_count=1
    for uri in etradis_category:
       base_uris = d.get_basuris(uri)
       batch = uri.replace("http://localhost:9999/etradis/", "")
       isEmpty = (base_uris == set())
       if isEmpty:
        print("Set is empty")
       else:
        line_count = 1
        for base_uri in base_uris:
            if base_uri in done_uris.keys():
                continue
            if "__" in base_uri:
                continue
            print(" total_count::"+str(total_count)+" now::"+str(line_count)+" base::"+base_uri+" batch::"+batch)
            total_count = total_count + line_count
            id = "_"+batch + "_"+str(line_count)
            print("DUMP files for nodes and connections")
            print("CRAWL data from DBPedia sparql endpoint")
            d.retrieve_data(start_uri=base_uri, search_depth=search_depth,
                            filter_threshold=3)
            node_path=os.path.join(data_path, f"nodes_d{id}.pkl")
            connections_path=os.path.join(data_path, f"connections_d{id}.pkl")
            with open(node_path, "wb") as nodes_file:
                pickle.dump(d.nodes, nodes_file)
            with open(connections_path, "wb") as connections_file:
                pickle.dump(d.connections, connections_file)
            line_count = line_count + 1
            print(str(total_count) + " " + node_path + " " + connections_path)
            insert_into_neo4j(do_cleanup, update_index, depth, data_path, node_path, connections_path, graph)
            #done_file.write(base_uri+","+batch+"\n")
            with open(done_uris_path, 'a') as outfile:
                wr = csv.writer(outfile, delimiter=',')
                wr.writerow([base_uri,batch])
        print("finish of category::"+batch+" total_count::"+str(total_count))


    print("finish all categories::")
    #done_file.close()

if __name__ == "__main__":
    main()


