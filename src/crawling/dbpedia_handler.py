#!/usr/bin/env python3

from collections import defaultdict
from time import sleep
from typing import DefaultDict, Set, Tuple, List

from easysparql import easysparql as es


class DBPediaHandler:

    def __init__(self) -> None:
        """
        Initialise DBPedia handler which handles crawling process of dbpedia content using sparql endpoint.
        """
        #self.sparql_endpoint = "http://dbpedia.org/sparql/"
        self.sparql_endpoint = "http://localhost:9999/blazegraph/sparql"
        self.max_retries = 3  # define number of retries for any sparql query
        self.sleep_time = 3  # define time in seconds to wait for retry
        # specify uris which should be handled as property for any node
        self.node_property_names = ["http://dbpedia.org/ontology/wikiPageExternalLink",
                                     "http://www.w3.org/1999/02/22-rdf-syntax-ns#type", "http://www.w3.org/2002/07/owl#sameAs"]
        self.nodes = {}
        self.connections = set()

    def _get_superclasses(self, uri: str) -> List:
        """
        Helper function to retrieve super classes (one level below own:Thing) for a specific uri according to dbpedia ontology schema (label tree)
        :param uri: specify uri of dbpedia resource
        :return: List contains super classes for given entry
        """
        super_classes = []
        query = f"""
               PREFIX dbr: <http://dbpedia.org/resource/>
               SELECT DISTINCT ?x
               WHERE {{
                   <{uri}> <http://localhost:9999/etradis#type> ?x .
               }}
               """
        try:
            results = es.run_query(query, self.sparql_endpoint)
            if results:
                for res in results:
                    str=res.get("x").get("value")
                    if str=="Miscellaneous":
                        continue
                    elif str=="OtherClass":
                        str="Miscellaneous"
                    super_classes.append(str)
            else:
                super_classes = ["owl:Thing"]
        except Exception as e:
            print(e)
        return super_classes

    def get_basuris(self, uri: str) -> List:
        """
        Helper function to retrieve base uris of specific categories
        :return: List contains base URIs super classes for given category
        """
        base_uris = set()

        query = f"""
              SELECT DISTINCT ?s 
              WHERE {{
                  ?s <http://localhost:9999/etradis#type>  <{uri}> .
              }}
              """

        try:
            results = es.run_query(query, self.sparql_endpoint)
            if results:
                for res in results:
                    str = res.get("s").get("value")
                    base_uris.add(str)
        except Exception as e:
            print(e)
        return base_uris


    def get_wiki_uris(self, uri: str) -> List:
        """
        Helper function to retrieve base uris of specific categories
        :return: List contains base URIs super classes for given category
        """
        str = ""

        query = f"""
              PREFIX wd: <http://www.wikidata.org/>
              SELECT DISTINCT  ?o
              WHERE {{
                <{uri}> <http://www.w3.org/2002/07/owl#sameAs> ?o .
              FILTER ( STRSTARTS(STR(?o), STR(wd:)))
             }}
              """

        try:
            results = es.run_query(query, self.sparql_endpoint)
            if results:
                for res in results:
                    str = res.get("o").get("value")
                    break
        except Exception as e:
            print(e)
        return str




    def _get_predicate_object(self, uri: str, node: DefaultDict, connections: Set) -> bool:
        """
        Helper function to retrieve outgoing properties and relations of a given dbpedia resource.
        :param uri: dbpedia resource identifier
        :param node: store properties
        :param connections: store connections to other resources
        :return: boolean indicates whether query succeeded
        """
        query = f"""
        PREFIX dbo: <http://dbpedia.org/ontology/>
        PREFIX dbr: <http://dbpedia.org/resource/>
        PREFIX dbt: <http://dbpedia.org/property/wikiPageUsesTemplate>
        PREFIX owl: <http://www.w3.org/2002/07/owl#>
        PREFIX wd: <http://www.wikidata.org/>
        SELECT DISTINCT ?p ?o
        WHERE {{
            <{uri}> ?p ?o .
            FILTER ( STRSTARTS(STR(?o), STR(dbo:)) || STRSTARTS(str(?o), STR(dbr:)) || isLiteral(?o) || STRSTARTS(STR(?p), STR(dbo:wikiPageExternalLink)) || (STRSTARTS(STR(?p), STR(owl:sameAs)) && (STRSTARTS(STR(?o), STR(wd:entity))))) .
            FILTER ( !STRSTARTS(STR(?p), STR(dbt:)) ) .
        }}
        """
        try:
            results = es.run_query(query, self.sparql_endpoint)
            for res in results:
                predicate_val = res.get("p").get("value")
                obj = res.get("o")
                # add as property to node or mark as connection between two nodes
                # if predicate_val in self.node_property_names or obj.get("type") != "uri":
                if predicate_val in self.node_property_names or obj.get("type") != "uri":
                    if "xml:lang" in obj:
                        if obj.get('xml:lang') == "en":
                            object_val = f"{obj.get('value')}@{obj.get('xml:lang')}"
                            node[predicate_val].append(object_val)
                    elif obj.get("datatype") == "http://www.w3.org/2001/XMLSchema#date":
                        date_prop = (predicate_val, obj.get("value"))
                        node["dates"].append(date_prop)
                    # elif predicate_val == "http://www.w3.org/1999/02/22-rdf-syntax-ns#type":
                    else:
                        node[predicate_val].append(obj.get("value"))
                else:
                    connections.add((uri, predicate_val, obj.get("value")))
            return True
        except Exception as e:
            print(e)
            return False

    def _get_predicate_subject(self, uri: str, node: DefaultDict, connections: Set) -> bool:
        """
        Helper function to retrieve incoming properties and relations of a given dbpedia resource
        :param uri: dbpedia resource identifier
        :param node: store properties
        :param connections: store connections to other resources
        :return: boolean indicates whether query succeeded
        """
        query = f"""
        PREFIX dbo: <http://dbpedia.org/ontology/>
        PREFIX dbr: <http://dbpedia.org/resource/>
        PREFIX dbt: <http://dbpedia.org/property/wikiPageUsesTemplate>
        SELECT DISTINCT ?p ?s
        WHERE {{
            ?s ?p <{uri}> .
            FILTER ( STRSTARTS(STR(?s), STR(dbo:)) || STRSTARTS(str(?s), STR(dbr:)) || isLiteral(?s) || STRSTARTS(STR(?p), STR(dbo:wikiPageExternalLink)) ) .
            FILTER ( !STRSTARTS(STR(?p), STR(dbt:)) ) .
        }}
        """
        try:
            results = es.run_query(query, self.sparql_endpoint)
            for res in results:
                predicate_val = res.get("p").get("value")
                subj = res.get("s")
                # add as property to node or mark as connection between two nodes
                if predicate_val in self.node_property_names or subj.get("type") != "uri":
                    if "xml:lang" in subj:
                        if subj.get('xml:lang') == "en":
                            subject_val = f"{subj.get('value')}@{subj.get('xml:lang')}"
                            node[predicate_val].append(subject_val)
                    elif subj.get("datatype") == "http://www.w3.org/2001/XMLSchema#date":
                        date_prop = (predicate_val, subj.get("value"))
                        node["dates"].append(date_prop)
                    # elif predicate_val == "http://www.w3.org/1999/02/22-rdf-syntax-ns#type":
                    else:
                        node[predicate_val].append(subj.get("value"))
                else:
                    connections.add((subj.get("value"), predicate_val, uri))
            return True
        except Exception as e:
            print(e)
            return False

    def _get_resource(self, uri: str) -> Tuple[DefaultDict, Set]:
        """
        Helper function to retrieve all properties and relations of a given dbpedia resource
        :param uri: dbpedia resource identifier
        :return: Tuple contains properties and set of corresponding connections
        """
        node = defaultdict(list)
        connections = set()
        for idx in range(self.max_retries):
            success = self._get_predicate_object(uri, node, connections)
            if success:
                break
            else:
                print("retry attempt ", idx + 1)
                sleep(self.sleep_time)
        for idx in range(self.max_retries):
            success = self._get_predicate_subject(uri, node, connections)
            if success:
                break
            else:
                print("retry attempt ", idx + 1)
                sleep(self.sleep_time)
        return node, connections

    def _resolve_connections(self,base_uri:str) -> None:
        """
        Helper function to update set of all connections between two nodes by adding nodes if not available.
        When creating connections in neo4j, source and target nodes have to be available to create a connection.
        :return:
        """
        subjIndex=1
        objIndex = 1
        new_connections = set()
        for con in self.connections:
            subj = con[0]
            obj = con[2]
            if subj == base_uri:
                subjIndex = subjIndex + 1
                # print(" subj::"+subj+" "+obj)
            elif obj == base_uri:
                objIndex = objIndex + 1
                # print("obj::"+subj + " " + obj+" "+base_uri)
            #if subjIndex>=500 and objIndex>=500:
            #    break
            #print("subjIndex::"+str(subjIndex)+" subj::"+subj+ "  obj::" + obj+ "  objIndex::"+str(objIndex) +"  property: "+con[1])
            if subj not in self.nodes:
                node, subject_connections = self._get_resource(subj)
                self.nodes[subj] = node
                new_connections.update(subject_connections)
            if obj not in self.nodes:
                node, object_connections = self._get_resource(obj)
                self.nodes[obj] = node
                new_connections.update(object_connections)
        self.connections.update(new_connections)

    def _remove_non_relevant_connections(self, threshold: int) -> None:
        """
        Helper function to remove nodes and corresponding connections of nodes with fewer connections then a given threshold
        :param threshold: specify minimum number of connections of any specific node
        :return:
        """
        node_relevance = {}
        for con in self.connections:
            subj = con[0]
            obj = con[2]
            if subj not in node_relevance:
                node_relevance[subj] = 0
            else:
                node_relevance[subj] += 1

            if obj not in node_relevance:
                node_relevance[obj] = 0
            else:
                node_relevance[obj] += 1
        to_remove = set()
        for k, v in node_relevance.items():
            if v <= threshold:
                to_remove.add(k)
        for node_id in list(to_remove):
            if node_id in self.nodes:
                self.nodes.pop(node_id)
        cleaned_connections = set()
        for con in self.connections:
            subj = con[0]
            obj = con[2]
            if subj not in to_remove and obj not in to_remove:
                cleaned_connections.add(con)
        self.connections = cleaned_connections

    def _remove_unresolved_connections(self) -> Set:
        """
        Helper function to remove connections where source or target node do not exist.
        :return: set contains cleaned connections which can be created in neo4j
        """
        resolved_connections = set()
        for con in self.connections:
            subj = con[0]
            obj = con[2]
            if subj in self.nodes and obj in self.nodes:
                resolved_connections.add(con)
        return resolved_connections

    def _filter_connections(self, threshold: int) -> None:
        """
        Helper function to reduce number of connection after each iteration of search depth.
        :param threshold: specify minimum number of connections of any specific node
        :return:
        """
        # number of connections is below given threshold
        self._remove_non_relevant_connections(threshold)
        self.connections = self._remove_unresolved_connections()

    def _map_classes(self, super_classes: List) -> str:
        """
        Helper function to reduce dbpedia owl super classes to a given set of available labels.
        :param super_classes: list of dbpedia owl super classes of a node
        :return:
        """
        # TODO: improve mapping by nlp, ml, etc...
        mapped_class = "Miscellaneous"
        for c in super_classes:
            if "event" in c.lower():
                mapped_class = "Event"
                break
            elif any(agent in c.lower() for agent in ["agent", "species", "ethnicgroup", "language"]):
                mapped_class = "Agent"
                break
            elif "place" in c.lower():
                mapped_class = "Place"
                break
            elif "timeperiod" in c.lower():
                mapped_class = "TimePeriod"
                break
            elif "topicalconcept" in c.lower():
                mapped_class = "TopicalConcept"
                break
            elif "work" in c.lower():
                mapped_class = "CulturalArtifact"
                break
            elif any(mat_obj in c.lower() for mat_obj in ["materialobject", "meanoftransportation", "currency", "device", "food", "chemicalsubstance"]):
                mapped_class = "MaterialObject"
                break
            else:
                mapped_class = "Miscellaneous"
        return mapped_class

    def get_mapped_class(self, uri) -> str:
        """
        Retrieve label for any given node.
        :param uri: dbpedia resource identifier
        :return: string contains mapped label
        """
        super_classes = []
        for idx in range(self.max_retries):
            super_classes = self._get_superclasses(uri)
            if super_classes:
                break
            else:
                print("retry attempt ", idx + 1)
                sleep(self.sleep_time)
        mapped_class = self._map_classes(super_classes)
        return mapped_class

    def get_original_class(self, uri) -> str:
        """
        Retrieve label for any given node.
        :param uri: dbpedia resource identifier
        :return: string contains mapped label
        """
        super_classes = []
        query = f"""
                      SELECT DISTINCT ?x
                      WHERE {{
                          <{uri}> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> ?x .
                      }}
                      """
        try:
            results = es.run_query(query, self.sparql_endpoint)
            if results:
                for res in results:
                    str = res.get("x").get("value")
                    super_classes.append(str)
            else:
                super_classes = ["owl:Thing"]
        except Exception as e:
            print(e)
        return super_classes

    def get_rdf_label(self, uri) -> str:
        query = f"""
                      SELECT DISTINCT ?x
                      WHERE {{
                          <http://dbpedia.org/resource/Razowskiina_psydra> <http://www.w3.org/2000/01/rdf-schema#label> ?x .
                      }}
                      """
        label=""
        try:
            results = es.run_query(query, self.sparql_endpoint)
            if results:
                for res in results:
                    label = res.get("x").get("value")
        except Exception as e:
            print(e)
        print(" uri::"+uri+" label::"+label)
        return label

    def retrieve_data(self, start_uri: str, search_depth: int, filter_threshold: int) -> None:
        """
        For a given entry URI crawl data from dbpedia which consists of entry node and adjacent nodes of a given depth.
        :param start_uri: entry uri
        :param search_depth: search depth starting from entry node
        :param filter_threshold: remove nodes with fewer connections
        :return:
        """
        if start_uri in self.nodes:
            return
        else:
            node, connections = self._get_resource(start_uri)
            self.nodes[start_uri] = node
            self.connections.update(connections)

            for idx in range(search_depth):
                print(f"depth: {idx}")
                print("number nodes before further resolving:", len(self.nodes))
                print("number connections before further resolving: ", len(self.connections))
                self._resolve_connections(start_uri)
                print("number nodes before filtering:", len(self.nodes))
                print("number connections before filtering: ", len(self.connections))
                self._remove_non_relevant_connections(filter_threshold)
                print("number nodes after filtering:", len(self.nodes))
                print("number connections after filtering: ", len(self.connections))

            print("\nfinal filter step")
            print("number nodes before filtering:", len(self.nodes))
            print("number connections before filtering: ", len(self.connections))

            self._filter_connections(filter_threshold)
            print("number nodes after filtering:", len(self.nodes))
            print("number connections after filtering: ", len(self.connections))
