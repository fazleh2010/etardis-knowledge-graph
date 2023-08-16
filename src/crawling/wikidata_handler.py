#!/usr/bin/env python3
import json
from collections import OrderedDict
from datetime import datetime
from typing import List, Dict

from easysparql import easysparql as es


class WikidataHandler:

    def __init__(self):
        """
        Initialise DBPedia handler which handles crawling process of wikidata content using sparql endpoint.
        """
        self.sparql_endpoint = "https://query.wikidata.org/sparql"
        self.placeholder = "unknown"

    def get_default_description(self, node_label: str) -> Dict:
        """
        Retrieve default description represented as dict which contains placeholders for expected entries depending on label of node.
        :param node_label: string contains label (type) of node
        :return: dict contains default description
        """
        description = {"period": self.placeholder, "periodPredicted": False, "locations": self.placeholder,
                       "locationsPredicted": False}
        if node_label == "Event":
            description["facets"] = self.placeholder
            description["participants"] = self.placeholder
        elif node_label == "Agent":
            description["genders"] = self.placeholder
            description["families"] = self.placeholder
            description["positions"] = self.placeholder
            description["religions"] = self.placeholder
        elif node_label == "TopicalConcept":
            description["facets"] = self.placeholder
            description["uses"] = self.placeholder
        elif node_label == "CulturalArtifact":
            description["authors"] = self.placeholder
        elif node_label == "MaterialObject":
            description["material"] = self.placeholder
        elif node_label == "Place":
            pass
        elif node_label == "TimePeriod":
            pass
        elif node_label == "Miscellaneous":
            pass
        return description

    def get_image(self, wiki_id: str) -> List:
        """
        Retrieve list of parsed images for a given wikidata identifier.
        :param wiki_id: wikidata identifier of node (dbpedia uri and wikidata identifier are both unique)
        :return: list contains image data
        """
        images = set()
        query = f"""
        SELECT ?image WHERE {{
            BIND ( wd:{wiki_id} AS ?id ).
            {{
                ?id wdt:P18 ?image.
            }}
        }}
        """
        try:
            results = es.run_query(query, self.sparql_endpoint)
            for res in results:
                if "image" in res:
                    # TODO: encode as base64
                    c = res.get("image").get("value")
                    images.add(c)
        except Exception as e:
            images = []
            print(e)
        return list(images)

    def get_locations(self, wiki_id: str) -> List:
        """
        Retrieve list of corresponding locations for a given wikidata identifier.
        :param wiki_id: wikidata identifier of node (dbpedia uri and wikidata identifier are both unique)
        :return: list consists of tuples of latitude and longitude
        """
        coordinates = self.placeholder
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
            results = es.run_query(query, self.sparql_endpoint)
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

    def get_period(self, wiki_id: str) -> List:
        """
        Retrieve list of corresponding periods for a given wikidata identifier.
        :param wiki_id: wikidata identifier of node (dbpedia uri and wikidata identifier are both unique)
        :return: list consists of minimum and maximum mentioned year which form the period
        """
        period = self.placeholder
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
            results = es.run_query(query, self.sparql_endpoint)
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

    def get_material_object(self, wiki_id: str) -> List:
        """
        Retrieve list of corresponding material objects for a given wikidata identifier.
        :param wiki_id: wikidata identifier of node (dbpedia uri and wikidata identifier are both unique)
        :return: list consists of strings of mentioned materials
        """
        materials = self.placeholder
        query = f"""
        SELECT ?material WHERE {{
            wd:{wiki_id} wdt:P186 ?m.
            ?m rdfs:label ?material.
            FILTER(((LANG(?material)) = '') || (LANGMATCHES(LANG(?material), 'en')))
        }}
        """
        try:
            tmp_materials = set()
            results = es.run_query(query, self.sparql_endpoint)
            for res in results:
                if "material" in res:
                    tmp_materials.add(res.get("material").get("value"))
            if tmp_materials:
                materials = sorted(list(tmp_materials))
        except Exception as e:
            materials = []
            print(e)
        return materials

    def get_cultural_artifact(self, wiki_id: str) -> List:
        """
        Retrieve list of corresponding cultural artifacts for a given wikidata identifier.
        :param wiki_id: wikidata identifier of node (dbpedia uri and wikidata identifier are both unique)
        :return: list consists of strings of mentioned authors
        """
        artifacts = self.placeholder
        query = f"""
        SELECT DISTINCT ?author WHERE {{
            BIND ( wd:{wiki_id} AS ?id ).
            {{
                ?id wdt:P50 ?a.
                ?a rdfs:label ?author .
            }}
            UNION
            {{
                ?id wdt:P170 ?a .
                ?a rdfs:label ?author .
            }}
            FILTER(((LANG(?author)) = '') || (LANGMATCHES(LANG(?author), 'en')))
        }}"""
        try:
            tmp_artifacts = set()
            results = es.run_query(query, self.sparql_endpoint)
            for res in results:
                if "author" in res:
                    tmp_artifacts.add(res.get("author").get("value"))
            if tmp_artifacts:
                artifacts = sorted(list(tmp_artifacts))
        except Exception as e:
            artifacts = []
            print(e)
        return artifacts

    def get_topical_concept(self, wiki_id: str) -> OrderedDict:
        """
        Retrieve dict of corresponding topical concept entries for a given wikidata identifier.
        :param wiki_id: wikidata identifier of node (dbpedia uri and wikidata identifier are both unique)
        :return: OrderedDict consists of facet and use entries each represented as list of strings
        """
        concepts = OrderedDict()
        concepts["facets"] = self.placeholder
        concepts["uses"] = self.placeholder
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
            results = es.run_query(query, self.sparql_endpoint)
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

    def get_agent(self, wiki_id: str) -> OrderedDict:
        """
        Retrieve dict of corresponding agent entries for a given wikidata identifier.
        :param wiki_id: wikidata identifier of node (dbpedia uri and wikidata identifier are both unique)
        :return: OrderedDict consists of mentioned gender, family, position and religion entries each represented as list of strings
        """
        agents = OrderedDict()
        agents["genders"] = self.placeholder
        agents["families"] = self.placeholder
        agents["positions"] = self.placeholder
        agents["religions"] = self.placeholder
        genders = set()
        families = set()
        positions = set()
        religions = set()
        query = f"""
            SELECT DISTINCT ?gender ?family ?position ?religion WHERE {{
                BIND ( wd:{wiki_id} AS ?id ).
                OPTIONAL {{
                    ?id wdt:P21 ?s.
                    ?s rdfs:label ?gender .
                    FILTER(((LANG(?gender)) = '') || (LANGMATCHES(LANG(?gender), 'en')))
                }}
                OPTIONAL {{
                    ?id wdt:P53 ?f .
                    ?f rdfs:label ?family .
                    FILTER(((LANG(?family)) = '') || (LANGMATCHES(LANG(?family), 'en')))
                }}
                OPTIONAL {{
                    ?id wdt:P39 ?p .
                    ?p rdfs:label ?position .
                    FILTER(((LANG(?position)) = '') || (LANGMATCHES(LANG(?position), 'en')))
                }}
                OPTIONAL {{
                    ?id wdt:P140 ?r .
                    ?r rdfs:label ?religion .
                    FILTER(((LANG(?religion)) = '') || (LANGMATCHES(LANG(?religion), 'en')))
                }}
            }}"""
        try:
            results = es.run_query(query, self.sparql_endpoint)
            for res in results:
                if "gender" in res:
                    genders.add(res.get("gender").get("value"))
                if "family" in res:
                    families.add(res.get("family").get("value"))
                if "position" in res:
                    positions.add(res.get("position").get("value"))
                if "religion" in res:
                    religions.add(res.get("religion").get("value"))
            if genders:
                agents["genders"] = list(genders)
            if families:
                agents["families"] = list(families)
            if positions:
                agents["positions"] = list(positions)
            if religions:
                agents["religions"] = list(religions)
        except Exception as e:
            agents = OrderedDict()
            print(e)
        return agents

    def get_event(self, wiki_id: str) -> OrderedDict:
        """
        Retrieve dict of corresponding event entries for a given wikidata identifier.
        :param wiki_id: wikidata identifier of node (dbpedia uri and wikidata identifier are both unique)
        :return: OrderedDict consists of facet and participant entries each represented as list of strings
        """
        events = OrderedDict()
        events["facets"] = self.placeholder
        events["participants"] = self.placeholder
        facets = set()
        participants = set()
        query = f"""
        SELECT DISTINCT ?facet ?participant WHERE {{
            BIND ( wd:{wiki_id} AS ?id ).
            OPTIONAL {{
                ?id wdt:P1269 ?f.
                ?f rdfs:label ?facet .
                FILTER(((LANG(?facet)) = '') || (LANGMATCHES(LANG(?facet), 'en')))
            }}
            OPTIONAL {{
                ?id wdt:P710 ?p .
                ?p rdfs:label ?participant .
                FILTER(((LANG(?participant)) = '') || (LANGMATCHES(LANG(?participant), 'en')))
            }}
        }}"""
        try:
            results = es.run_query(query, self.sparql_endpoint)
            for res in results:
                if "facet" in res:
                    facets.add(res.get("facet").get("value"))
                if "participant" in res:
                    participants.add(res.get("participant").get("value"))
            if facets:
                events["facets"] = list(facets)
            if participants:
                events["participants"] = list(participants)
        except Exception as e:
            events = OrderedDict()
            print(e)
        return events

    def get_time_period(self, wiki_id: str):
        """
        Retrieve time period for a given wikidata identifier.
        :param wiki_id: wikidata identifier of node (dbpedia uri and wikidata identifier are both unique)
        :return:
        """
        # TODO: implement or remove
        pass

    def get_place(self, wiki_id: str):
        """
        Retrieve places for a given wikidata identifier.
        :param wiki_id: wikidata identifier of node (dbpedia uri and wikidata identifier are both unique)
        :return:
        """
        # TODO: implement or remove
        pass

    def get_miscellaneous(self, wiki_id: str):
        """
        Retrieve further values for a given wikidata identifier.
        :param wiki_id: wikidata identifier of node (dbpedia uri and wikidata identifier are both unique)
        :return:
        """
        # TODO: implement or remove
        pass
