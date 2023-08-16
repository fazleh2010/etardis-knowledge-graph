#!/usr/bin/env python3

import ast
import csv
import os
import pickle
from collections import defaultdict, Counter
from typing import Dict, List

from anytree import Node
from anytree.exporter import DotExporter, JsonExporter
from easysparql import easysparql as es
from wordcloud import WordCloud

from src.crawling.graph_handler2 import GraphHandler


def _extract_pickle(file: str):
    with open(file, "rb") as infile:
        return pickle.load(infile)


# def _check_labels(node: Dict, determined_labels: List):
#     unknown_assignments = 0
#     for _, v in nodes.items():
#         node_labels = v.get("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
#         contains = False
#         if isinstance(node_labels, List):
#             for lbl in determined_labels:
#                 if lbl in node_labels:
#                     contains = True
#         else:
#             if node_labels:
#                 contains = node_labels in determined_labels
#         if not contains:
#             unknown_assignments += 1
#             print(node_labels)
#
#     print(unknown_assignments)


def _reformat_label(label: str) -> str:
    if label:
        return label.replace("http://dbpedia.org/resource/", "").replace("http://dbpedia.org/ontology/", "").strip()
    else:
        return ""


def _reformat_connection(label: str) -> str:
    if "http://dbpedia.org/ontology/" in label:
        label = label.replace("http://dbpedia.org/ontology/", "")
    elif "http://dbpedia.org/resource/" in label:
        label = label.replace("http://dbpedia.org/resource/", "")
    elif "owl:Thing" in label:
        label = "Thing"
    return f"has{label.capitalize()}"


def count_node_relations(connections: List, output_path: str):
    count_connections = {}
    for con in connections:
        sub = _reformat_label(con[0])
        obj = _reformat_label(con[2])

        if sub not in count_connections:
            count_connections[sub] = 1
        else:
            count_connections[sub] += 1

        if obj not in count_connections:
            count_connections[obj] = 1
        else:
            count_connections[obj] += 1

    wc = WordCloud(width=1920, height=1080, max_font_size=200, max_words=200, background_color="white",
                   collocations=False)
    wc.generate_from_frequencies(count_connections)

    # store to file
    wc.to_file(os.path.join(output_path, "nodes.png"))


def investigate_labels(nodes: Dict, updated_labels: Dict, output_path: str) -> List:
    count_labels = dict()
    for k, _ in nodes.items():
        node_labels = updated_labels[k]
        for lbl in node_labels:
            lbl = _reformat_label(lbl)
            if lbl not in count_labels:
                count_labels[lbl] = 1
            else:
                count_labels[_reformat_label(lbl)] += 1

    # Create a word cloud image
    wc = WordCloud(width=1920, height=1080, max_font_size=300, max_words=30, background_color="white",
                   collocations=False)
    wc.generate_from_frequencies(count_labels)

    # store to file
    wc.to_file(os.path.join(output_path, "labels.png"))

    return sorted(count_labels.items(), key=lambda x: x[1], reverse=True)[:10]


def relevance_of_connections(connections: List, updated_labels: Dict, output_path: str):
    connection_appearance = {}
    for con in connections:
        _, rel_property, obj = con
        for obj_label in updated_labels[obj]:
            rel_name = _reformat_connection(obj_label)
            if rel_name not in connection_appearance:
                connection_appearance[rel_name] = 1
            else:
                connection_appearance[rel_name] += 1

    wc = WordCloud(width=1920, height=1080, min_font_size=20, max_font_size=300, max_words=30, background_color="white",
                   collocations=False)
    wc.generate_from_frequencies(connection_appearance)

    # store to file
    wc.to_file(os.path.join(output_path, "connections.png"))


def _determine_subclass_hierarchy(uri: str):
    sub_classes = defaultdict(list)
    lang_regex = "http://dbpedia.org/ontology/[0-9a-zA-Z]"  # used to ignore arabic labels
    query = f"""
    PREFIX dbo: <http://dbpedia.org/ontology/>
    PREFIX dbr: <http://dbpedia.org/resource/>
    PREFIX dbt: <http://dbpedia.org/property/wikiPageUsesTemplate>   
    PREFIX owl: <http://www.w3.org/2002/07/owl#>
    SELECT DISTINCT ?x ?o
    WHERE {{
        ?o rdfs:subClassOf* <{uri}> .
        OPTIONAL {{ ?o rdfs:subClassOf ?x }} .
        FILTER( !STRSTARTS(STR(?x), STR(owl:) )) .
        FILTER( STRSTARTS(STR(?x), STR(dbo:)) || STRSTARTS(str(?x), STR(dbr:)) ) .
        FILTER( REGEX(STR(?o), "{lang_regex}") ) .
        FILTER( REGEX(STR(?x), "{lang_regex}") ) .
    }}
    """
    try:
        results = es.run_query(query, "http://dbpedia.org/sparql/")
        if results:
            for res in results:
                parent = res.get("x").get("value")
                child = res.get("o").get("value")
                sub_classes[parent].append(child)
    except:
        pass
    return sub_classes


def generate_label_tree(updated_labels: Dict, output_path: str):
    top_labels = set()

    for _, labels in updated_labels.items():
        for lbl in labels:
            if lbl != "owl:Thing":
                top_labels.add(lbl)

    nodes_lookup = {}
    root = Node("owl:Thing")
    nodes_lookup["root"] = root

    for lbl in list(top_labels):
        sub_classes = _determine_subclass_hierarchy(lbl)
        nodes_lookup[_reformat_label(lbl)] = Node(_reformat_label(lbl), parent=nodes_lookup["root"])
        node_tuples = []
        for k, vals in sub_classes.items():
            for v in vals:
                name = _reformat_label(v)
                parent = _reformat_label(k)
                node_tuples.append((name, parent))
        while node_tuples:
            insert_later = []
            for idx, tup in enumerate(node_tuples):
                try:
                    nodes_lookup[tup[0]] = Node(tup[0], parent=nodes_lookup[tup[1]])
                except Exception as e:
                    insert_later.append(tup)
            node_tuples = insert_later
    # save ontology as json
    json_representation = JsonExporter(indent=4, sort_keys=True, ensure_ascii=False).export(root)
    with open(os.path.join(output_path, "label_ontology.json"), "w") as outfile:
        outfile.write(json_representation)
    # save ontology as pdf
    DotExporter(root, graph="digraph").to_picture(os.path.join(output_path, "label_ontology.pdf"))


def check_property_availability(property: str) -> Dict:
    url = "bolt://localhost:7687"
    username = "neo4j"
    password = "password"
    g = GraphHandler(url, username, password)

    unknown_nodes = []
    query = f"""
        MATCH (n) 
        WHERE n.{property} = "unknown" 
        RETURN n.uri AS uri
    """
    results = g.graph.run(query)
    for res in results:
        unknown_nodes.append(res.get("uri"))

    nodes_unknown_prop = len(unknown_nodes)
    unpredictable_nodes = nodes_unknown_prop
    for uri in unknown_nodes:
        query = f"""
            MATCH (n)-[r]-(m) 
            WHERE n.uri = "{uri}" AND m.{property} <> "unknown" 
            RETURN COUNT(m) AS cnt
        """
        res = g.graph.evaluate(query)
        if res > 0:
            unpredictable_nodes -= 1

    return {"unknown": nodes_unknown_prop, "unpredictable": unpredictable_nodes}




# nodes_path = "/data/nodes_d1.pkl"
# connections_path = "/data/connections_d1.pkl"
# updated_labels_path = "/data/updated_labels.csv"
# output_path = "/data/"
#
# updated_labels_lookup = dict()
# with open(updated_labels_path, "r") as infile:
#     reader = csv.reader(infile, delimiter=",")
#     for row in reader:
#         uri = row[0]
#         labels = ast.literal_eval(row[1])
#         updated_labels_lookup[uri] = labels
#
# nodes = _extract_pickle(nodes_path)
# connections = _extract_pickle(connections_path)

# count_node_relations(connections, output_path)
# investigate_labels(nodes, updated_labels_lookup, output_path)
# relevance_of_connections(connections, updated_labels_lookup, output_path)

# generate_label_tree(updated_labels_lookup, output_path)


print(check_property_availability("period"))
