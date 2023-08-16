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

    def extract_pickle(self, file: str):
        """
        Function to extract pickled datafiles.
        :param file: string which consists of given file path
        :return: extracted object
        """
        with open(file, "rb") as infile:
            return pickle.load(infile)

    def _format_uri(self, uri: str) -> str:
        """
        Helper function to reformat representation of dbpedia uris.
        :param uri: dbpedia resource identifier
        :return: string contains reformatted uri representation
        """
        # TODO: remove this step - reformatting should only happen in API
        if "http://dbpedia.org/ontology/" in uri:
            uri = uri.replace("http://dbpedia.org/ontology/", "dbo:")
        elif "http://dbpedia.org/resource/" in uri:
            uri = uri.replace("http://dbpedia.org/resource/", "dbr:")
        elif "http://dbpedia.org/property/" in uri:
            uri = uri.replace("http://dbpedia.org/property/", "dbp:")
        elif "http://www.w3.org/2000/01/rdf-schema#" in uri:  # check if comment still works
            uri = uri.replace("http://www.w3.org/2000/01/rdf-schema#", "w3:")
        elif "http://www.w3.org/2002/07/owl#" in uri:
            uri = uri.replace("http://www.w3.org/2002/07/owl#", "w3:")
        elif "http://purl.org/dc/terms/" in uri:
            uri = uri.replace("http://purl.org/dc/terms/", "purl:")
        elif "http://purl.org/linguistics/gold/" in uri:
            uri = uri.replace("http://purl.org/linguistics/gold/", "purl:")
        return uri.strip()

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

    def _retrieve_node_labels(self, labels: List) -> List:
        """
        Helper function to reformat label representation.
        :param labels: list of plain labels (dbpedia representation)
        :return: list contains reformatted label representations
        """
        if labels:
            formatted_labels = []
            for lbl in set(labels):
                formatted_labels.append(self._format_uri(lbl))
            return sorted(formatted_labels)
        else:
            return []

    def _extract_labels(self, nodes: Dict, updated_labels_path: str) -> List:
        """
        Helper function creates and stores a lookup table which is used to assign one of the available super classes to each node.
        :param nodes: Dict contains available nodes (key is uri, value is node content)
        :param updated_labels_path: string contains path where lookup table is stored
        :return: list of all available labels
        """
        all_labels = set()
        updated_labels_lookup = dict()
        for idx, (k, _) in enumerate(nodes.items()):
            current_label = self.dbp_handler.get_mapped_class(k)
            updated_labels_lookup[k] = current_label
            all_labels.add(current_label)
            if not idx % 100:
                print("Extracted updated labels for nodes: ", idx)
                print()
        with open(updated_labels_path, 'a') as outfile:
            wr = csv.writer(outfile, delimiter=',')
            for k, v in updated_labels_lookup.items():
                wr.writerow([k, v])
                #print("k::"+k+" v::"+v+" "+updated_labels_path)#
        return list(all_labels)

    def _create_relation_name(self, label: str) -> str:
        """
        Helper function to reformat name of relation.
        :param label: dbpedia representation of relation name
        :return: string contains reformatted relation name representation
        """
        # if "http://dbpedia.org/ontology/" in label:
        #     label = label.replace("http://dbpedia.org/ontology/", "")
        # elif "http://dbpedia.org/resource/" in label:
        #     label = label.replace("http://dbpedia.org/resource/", "")
        # elif "owl:Thing" in label:
        #     label = "Thing"
        return f"has{label.capitalize()}"

    def create_index(self, nodes: Dict, updated_labels_path: str) -> None:
        """
        Create uri indices for each available label to improve search speed in neo4j database.
        :param nodes: Dict contains available nodes (key is uri, value is node content)
        :param updated_labels_path: string contains path where lookup table is stored
        :return:
        """
        labels = self._extract_labels(nodes, updated_labels_path)
        for label in labels:
            try:
                if not self.graph.schema.get_indexes(label):
                    self.graph.schema.create_index(label, "uri")
            except Exception as e:
                print(f"Could not create label {label} due to exception {e}")

    def _create_basic_description(self, wiki_id: str) -> Tuple:
        """
        Helper function to create period and locations entries which are part of each node description and stored as properties.
        :param wiki_id: wikidata identifier of node (dbpedia uri and wikidata identifier are both unique)
        :return: tuple contains period and locations entries
        """
        period = self.wiki_handler.get_period(wiki_id)
        if not period:
            for idx in range(self.max_retries):
                period = self.wiki_handler.get_period(wiki_id)
                if period:
                    break
                else:
                    print("retry attempt ", idx + 1)
                    sleep(self.sleep_time)
            if not period:
                period = self.wiki_handler.placeholder
        locations = self.wiki_handler.get_locations(wiki_id)
        if not locations:
            for idx in range(self.max_retries):
                locations = self.wiki_handler.get_locations(wiki_id)
                if locations:
                    break
                else:
                    print("retry attempt ", idx + 1)
                    sleep(self.sleep_time)
            if not locations:
                locations = self.wiki_handler.placeholder
        return period, locations

    def _create_description(self, label: str, wiki_id: str) -> OrderedDict:
        """
        Helper function to create description part of a node which is taken from wikidata.
        :param label: reduced representation of super class name
        :param wiki_id: wikidata identifier of node (dbpedia uri and wikidata identifier are both unique)
        :return: dict contains key-value pairs for description entries which depends on type of node
        """
        # TODO: update description process by considering dbpedia and wikidata entries
        period, locations = self._create_basic_description(wiki_id)
        description = OrderedDict()
        description["period"] = period
        description["periodPredicted"] = False
        description["locations"] = locations
        description["locationsPredicted"] = False
        if label == "Event":
            events = self.wiki_handler.get_event(wiki_id)
            if not events:
                for idx in range(self.max_retries):
                    events = self.wiki_handler.get_event(wiki_id)
                    if events:
                        break
                    else:
                        print("retry attempt ", idx)
                        sleep(self.sleep_time)
            #description |= events
            description = {**description, **events}
        elif label == "Agent":
            agents = self.wiki_handler.get_agent(wiki_id)
            if not agents:
                for idx in range(self.max_retries):
                    agents = self.wiki_handler.get_agent(wiki_id)
                    if agents:
                        break
                    else:
                        print("retry attempt ", idx)
                        sleep(self.sleep_time)
            #description |= agents
            description = {**description, **agents}
        elif label == "Place":
            pass
            # self.wiki_handler.get_place(wiki_id)
        elif label == "TimePeriod":
            pass
            # self.wiki_handler.get_time_period(wiki_id)
        elif label == "TopicalConcept":
            concepts = self.wiki_handler.get_topical_concept(wiki_id)
            if not concepts:
                for idx in range(self.max_retries):
                    concepts = self.wiki_handler.get_topical_concept(wiki_id)
                    if concepts:
                        break
                    else:
                        print("retry attempt ", idx)
                        sleep(self.sleep_time)
            #description |= concepts
            description = {**description, **concepts}
        elif label == "CulturalArtifact":
            authors = self.wiki_handler.get_cultural_artifact(wiki_id)
            if not authors:
                for idx in range(self.max_retries):
                    authors = self.wiki_handler.get_cultural_artifact(wiki_id)
                    if authors:
                        break
                    else:
                        print("retry attempt ", idx)
                        sleep(self.sleep_time)
            description["authors"] = authors
        elif label == "MaterialObject":
            material = self.wiki_handler.get_material_object(wiki_id)
            if not material:
                for idx in range(self.max_retries):
                    material = self.wiki_handler.get_material_object(wiki_id)
                    if material:
                        break
                    else:
                        print("retry attempt ", idx)
                        sleep(self.sleep_time)
            description["material"] = material
        elif label == "Miscellaneous":
            pass
            # self.wiki_handler.get_miscellaneous(wiki_id)
        return description

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

                # create description using wikidata information
                wikidata_id = v.get("http://www.w3.org/2002/07/owl#sameAs")
                #print("wikidata_id::"+str(wikidata_id))
                if wikidata_id:
                    if isinstance(wikidata_id, list):
                        wikidata_id = str(wikidata_id[0]).replace("http://www.wikidata.org/entity/", "")
                        description = self._create_description(node_label, wikidata_id)
                        for label, content in description.items():
                            properties[label] = content
                    else:
                        print(wikidata_id)
                        properties = {**properties, **self.wiki_handler.get_default_description(node_label)}
                else:
                    # set empty placeholders for description entries
                    properties = {**properties, **self.wiki_handler.get_default_description(node_label)}
                if properties["period"] == "unknown":
                    if v["dates"]:
                        dates = v.pop("dates")
                        try:
                            collected_dates = list(
                                set([datetime.strptime(tup[1], '%Y-%m-%d').year for tup in dates]))
                            properties["period"] = [min(collected_dates), max(collected_dates)]
                            properties["periodPredicted"] = True
                        except Exception as e:
                            print(e)
                # detail content stored as json
                detail_content = OrderedDict()
                for prop_name, prop_val in v.items():
                    if prop_name == "dates":
                        pass
                    else:
                        detail_content[self._format_uri(prop_name)] = prop_val
                properties = {**properties, **detail_content}
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
            obj_label = self.dbp_handler.get_mapped_class(obj)
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