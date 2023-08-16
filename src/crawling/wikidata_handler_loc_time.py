#!/usr/bin/env python3
import json
import csv


from collections import OrderedDict
from datetime import datetime
from typing import List, Dict
from src.crawling.dbpedia_handler import DBPediaHandler
from easysparql import easysparql as es

sparql_endpoint= "https://query.wikidata.org/sparql"
wiki_id="Q22686"
placeholder = "unknown"

def get_period(wiki_id: str) -> List:
        period = placeholder
        times = set()
        query = f"""
        SELECT ?time WHERE {{
            BIND ( wd:{wiki_id} AS ?id ).
            {{
                ?id wdt:P580 ?time.
            }}
            UNION
            {{
                ?id wdt:P582 ?time.
            }}
            UNION
            {{
                ?id wdt:P2348 ?t1.
                OPTIONAL {{ ?t1 wdt:P580 ?time. }}
                OPTIONAL {{ ?t1 wdt:P582 ?time. }}
            }}
            UNION
            {{
                ?id wdt:P577 ?time. 
                ?id wdt:P577 ?time. 
            }}
            UNION
            {{ 
                ?id wdt:P585 ?time. 
                ?id wdt:P585 ?time. 
            }}
            UNION
            {{
                ?id wdt:P569 ?time.
            }}
            UNION
            {{
                ?id wdt:P570 ?time.
            }}
        }}"""
        try:
            results = es.run_query(query, sparql_endpoint)
            for res in results:
                if "time" in res:
                    time = res.get("time").get("value")
                    times.add(datetime.strptime(time, "%Y-%m-%dT%H:%M:%SZ").year)
            if times:
                period = [min(times), max(times)]
        except Exception as e:
            period = []
            print(e)
        return period

def get_locations(wiki_id: str) -> List:
        """
        Retrieve list of corresponding locations for a given wikidata identifier.
        :param wiki_id: wikidata identifier of node (dbpedia uri and wikidata identifier are both unique)
        :return: list consists of tuples of latitude and longitude
        """
        coordinates = placeholder
        query = f"""
        SELECT ?coordinates WHERE {{
            BIND ( wd:{wiki_id} AS ?id ).
            {{ 
                ?id wdt:P625 ?coordinates. 
            }}
            UNION
            {{
                ?id wdt:P276 ?location.
                ?location wdt:P625 ?coordinates.
            }}
            UNION
            {{
                ?id wdt:P17 ?country.
                ?country wdt:P625 ?coordinates.
            }}
            UNION
            {{
                ?id wdt:P495 ?originCountry.
                ?originCountry wdt:P625 ?coordinates.
            }}
            UNION
            {{
                ?id wdt:P840 ?narrativeLocation.
                ?narrativeLocation wdt:P625 ?coordinates.
            }}
            UNION
            {{
                ?id wdt:P27 ?countryCitizenship.
                ?countryCitizenship wdt:P625 ?coordinates.
            }}
            UNION
            {{
                ?id wdt:P19 ?placeBirth.
                ?placeBirth wdt:P625 ?coordinates.
            }}
            UNION
            {{
                ?id wdt:P20 ?placeDeath.
                ?placeDeath wdt:P625 ?coordinates.
            }}
        }}"""
        try:
            results = es.run_query(query, sparql_endpoint)
            tmp_coordinates = set()
            for res in results:
                if "coordinates" in res:
                    c = res.get("coordinates").get("value")
                    tmp_coordinates.add(tuple(c.replace("Point(", "").replace(")", "").split(" ")))
            if tmp_coordinates:
                coordinates = json.dumps(list(tmp_coordinates))
        except Exception as e:
            coordinates = []
            print(e)
        return coordinates

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

def main(uri_set=None):

    d = DBPediaHandler()
    data_path = "/media/elahi/Elements/A-Projects/etradis/resources/dataT/"
    done_uris_path = data_path + "wikidata_done_uris.csv"
    done_uris = get_categories(done_uris_path)
    etradis_category = ["http://localhost:9999/etradis/Agent"]

    updated_labels_lookup = dict()

    for uri in etradis_category:
       base_uris = d.get_basuris(uri)
       node_label = uri.replace("http://localhost:9999/etradis/", "")
       isEmpty = (base_uris == set())
       if isEmpty:
        print("dict is empty")
       else:
        line_count = 1
        for base_uri in base_uris:
            if base_uri in done_uris.keys():
                continue
            if "__" in base_uri:
                continue
            else :
                line_count=line_count+1
                wiki_uri = d.get_wiki_uris(uri)
                wikidata_id = str(wiki_uri).replace("http://www.wikidata.org/entity/", "")
                period = []
                location = []
                #period = get_period(wiki_id=wikidata_id)
                #location = get_locations(wiki_id=wikidata_id)
                print(str(line_count)+" base_uri::"+base_uri+" wiki_uri::"+wiki_uri+" wikidata_id::"+wikidata_id+" period:"+str(period) + " location::" + str(location)+" node_label::"+node_label)


    """
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
                print(nodes_path + " " + connections_path + " " + str(len(updated_labels_lookup)))
   
    period=[]
    location=[]
    period=get_period(wiki_id="Q22686")
    location=get_locations(wiki_id="Q22686")
    print(str(period)+" "+str(location))

  
    super_classes=[]
    str = "Miscellaneous"
    if str == "Miscellaneous":
        print(str)
    elif str == "OtherClass":
        str = "Miscellaneous"
    super_classes.append(str)
    print(super_classes)
    """

if __name__ == "__main__":
    main()