#!/usr/bin/env python3
import json
import os
import pickle


from collections import OrderedDict
from src.crawling.dbpedia_handler import DBPediaHandler

from datetime import datetime
from typing import List, Dict


from easysparql import easysparql as es

sparql_endpoint= "https://query.wikidata.org/sparql"
wiki_id="Q22686"
placeholder = "unknown"


def extract_pickle(file: str):
    """
    Function to extract pickled datafiles.
    :param file: string which consists of given file path
    :return: extracted object
    """
    with open(file, "rb") as infile:
        return pickle.load(infile)
def get_topical_concept(wiki_id: str) -> OrderedDict:
    """
    Retrieve dict of corresponding topical concept entries for a given wikidata identifier.
    :param wiki_id: wikidata identifier of node (dbpedia uri and wikidata identifier are both unique)
    :return: OrderedDict consists of facet and use entries each represented as list of strings
    """
    concepts = OrderedDict()
    concepts["facets"] = placeholder
    concepts["uses"] = placeholder
    facets = set()
    uses = set()
    query = f"""
        SELECT DISTINCT ?facet ?use WHERE {{
            BIND ( wd:{wiki_id} AS ?id ).
            OPTIONAL {{
                ?id wdt:P1269 ?f.
                ?f rdfs:label ?facet .
                FILTER(((LANG(?facet)) = '') || (LANGMATCHES(LANG(?facet), 'en')))
            }}
            OPTIONAL {{
                ?id wdt:P366 ?u .
                ?u rdfs:label ?use .
                FILTER(((LANG(?use)) = '') || (LANGMATCHES(LANG(?use), 'en')))
            }}
        }}"""
    try:
        results = es.run_query(query, sparql_endpoint)
        for res in results:
            if "facet" in res:
                facets.add(res.get("facet").get("value"))
            if "use" in res:
                uses.add(res.get("use").get("value"))
        if facets:
            concepts["facets"] = list(facets)
        if uses:
            concepts["uses"] = list(uses)
    except Exception as e:
        concepts = OrderedDict()
        print(e)
    return concepts


def main(uri_set=None):

    data_path = "/media/elahi/Elements/A-Projects/etradis/resources/data/"
    etradis_category = ["http://localhost:9999/etradis/TopicalConcept"]
    d = DBPediaHandler()

    updated_labels_lookup = dict()
    for uri in etradis_category:
        base_uris = d.get_basuris(uri)
        batch = uri.replace("http://localhost:9999/etradis/", "")
        isEmpty = (base_uris == set())
        if isEmpty:
            print("Set is empty")
        else:
            line_count = 1
            for base_uri in base_uris:
                # if base_uri in done_uris.keys():
                #    continue
                if "__" in base_uri:
                    continue
                else :
                    print( " base::" + base_uri + " batch::" + batch)

        """
        for file in os.listdir(data_path):
            nodeMat = "nodes_d_" + batch + "_"
            konMat = "connections_d_" + batch + "_"
            if file.startswith(nodeMat):
                nodes_path = os.path.join(data_path, file)
                strFile = file.replace(data_path, "")
                strFile = file.replace(nodeMat, konMat)
                connections_path = os.path.join(data_path, strFile)
                print(nodes_path + " " + connections_path + " " + str(len(updated_labels_lookup)))

    period=[]
    location=[]
    period=get_period(wiki_id="Q22686")
    location=get_locations(wiki_id="Q22686")
    print(str(period)+" "+str(location))
     """
    concepts = OrderedDict()
    concepts = get_topical_concept(wiki_id="Q22686")
    print(concepts)

if __name__ == "__main__":
    main()