#!/usr/bin/env python3

import csv
import pickle
from collections import OrderedDict
from datetime import datetime
from time import sleep
from typing import Dict, List, Set, Tuple

from py2neo import Graph, NodeMatcher
from py2neo.data import Node, Relationship

from src.crawling.dbpedia_handler import DBPediaHandler
from src.crawling.wikidata_handler import WikidataHandler


class GraphHandler:

    def __init__(self, url: str, username: str, password: str) -> None:
        """
        Initialise Neo4j graph handler which handles access to neo4j database.
        :param url: url to access neo4j graph instance
        :param username: username to access neo4j graph instance
        :param password: password to access neo4j graph instance
        """
        self.dbp_handler = DBPediaHandler()
        self.wiki_handler = WikidataHandler()
        self.placeholder = "unknown"
        self.max_retries = 3
        self.sleep_time = 3
        try:
            self.graph = Graph(url, auth=(username, password))
            self.node_matcher = NodeMatcher(self.graph)
        except Exception as e:
            raise RuntimeError(e)



    def delete_graph(self):
        """
        Function to delete all nodes and connections in neo4j graph database.
        Furthermore, the indices on uri for each available label are removed.
        :return:
        """
        for label in self.graph.schema.node_labels:
            indexes = self.graph.schema.get_indexes(label)
            for prop in indexes:
                self.graph.schema.drop_index(label, prop[0])
        self.graph.delete_all()


    def insert_nodes(self, nodes: Dict) -> None:
        """
        Function to insert nodes into neo4j database using transaction.
        :param nodes: Dict contains available nodes (key is uri, value is node content)
        :param updated_labels: dict contains lookup table to assign proper super class for each node
        :return:
        """
        updated_labels=dict()
        transaction = self.graph.begin()
        for idx, (k, v) in enumerate(nodes.items()):
            node = Node(uri=k)
            properties = OrderedDict()
            if not self.node_matcher.match(uri=k).exists():
                ## node_label = updated_labels[k]
                node_label = self.dbp_handler.get_mapped_class(k)
                updated_labels[k]=node_label
                node.add_label(node_label)
                #print("idx::" + str(idx) + " k:" + str(k) + " label::" + str(node_label))
                # handle name
                if "http://www.w3.org/2000/01/rdf-schema#label" in v:
                    name = v.pop("http://www.w3.org/2000/01/rdf-schema#label")
                else:
                    name = ""
                properties["name"] = name
                # handle subclasses
                if "http://www.w3.org/1999/02/22-rdf-syntax-ns#type" in v:
                    lbls = v.pop("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
                else:
                    lbls = []
                properties["subclasses"] = self._retrieve_node_labels(lbls)
                node.update(properties)
                transaction.create(node)
            if not idx % 100:
                print("Added nodes: ", idx)
                print()
        print("commit transaction")
        self.graph.commit(tx=transaction)


    def insert_connections(self, connections: Set) -> None:
        """
        Function to insert relations between available nodes into neo4j database using transaction.
        :param connections: set contains available connections between all given nodes
        :param updated_labels: dict contains lookup table to assign proper super class for each node
        :return:
        """
        index=1
        transaction = self.graph.begin()
        for idx, con in enumerate(list(connections)):
            subj, rel_property, obj = con
            node_n = self.node_matcher.match(uri=subj).first()
            node_m = self.node_matcher.match(uri=obj).first()
            # previous: add relation for each node type separately; now single label
            #obj_label = updated_labels[obj]
            obj_label = self.dbp_handler.get_mapped_class(obj)
            #relation_name = self._create_relation_name(obj_label)
            relation_name =f"has{obj_label.capitalize()}"
            index=index+1
            #print("idx:" + str(idx) + "obj:" + str(obj) + "  objLevel:" + str(obj_label)+" relation_name::"+str(relation_name)+" number_of_connections::"+str(len(con)))
            #if relation_name.startswith("hasMiscellaneous"):
            #    continue
            #if not self.graph.match(nodes=[node_m, node_n], r_type=relation_name).exists():
            nm_relationship = Relationship(node_n, relation_name, node_m)
            nm_relationship["rdf:Property"] = rel_property
            transaction.create(nm_relationship)
            if not idx % 1000:
             print("Added connections: ", str(idx)+" number_of_connections::"+str(len(connections))+" index::"+str(index)+" con::"+str(len(con)))
             print()
            #if index > 1000:
            #    break
        print("commit transaction")
        self.graph.commit(tx=transaction)

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
        graph = GraphHandler(url, username, password)
        updated_labels_path = os.path.join(data_path, "updated_labels.csv")

        etradis_category = ["http://localhost:9999/etradis/Event"]

        updated_labels_lookup = dict()
        index = 0
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
                    index = index + 1
                    print(str(index) + " " + nodes_path + " " + connections_path)
                    insert_into_neo4j(do_cleanup, update_index, depth, data_path, nodes_path, connections_path, graph)
                    exit(1)

    if __name__ == "__main__":
        main()