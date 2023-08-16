import ast
import json
import logging
# import time
import os
import codecs
from collections import OrderedDict
from typing import Dict, List, Tuple

import pickle

import math

# new imported libs for geo filter (12/2022)
####################################################################
# this script(s) is build on the code of https://github.com/che0/countries
# Description: Tools to find out Continent, country where given GPS coordinates are
# the shapefile with world borders is from 2009 by Bjørn Sandvik
# (https://thematicmapping.org/downloads/world_borders.php)
# Used in Functions:
# in functions (and their subfunctions) that contain 'GeoFilterByName' in name
# needs gdal to be installed to work with python 3.9. -> see requirements
####################################################################
from osgeo import ogr

from flask import Flask
from flask_cors import CORS
from flask_restx import Api, Resource, reqparse, fields
from py2neo import Graph, NodeMatcher

#import for semantic distance measures
import numpy as np
import ot  # calculation of emd

# import spacy
# https://github.com/explosion/spaCy/blob/master/LICENSE
# Evaluation of NER https://spacy.io/usage/facts-figures
from spacy.lang.en.stop_words import STOP_WORDS
# print(STOP_WORDS)
import textacy
from textacy import preprocessing as textacy_preprocessing
from scipy.spatial.distance import cosine

from sentence_transformers import SentenceTransformer
# look here for pretrained sBERT models: https://www.sbert.net/docs/pretrained_models.html

from gensim.corpora import Dictionary

class TextPreprocessing:

    def __init__(self, sentence_embedding_model="./fine_tuned_model_all-MiniLM-L6-v2"):
        self.en = textacy.load_spacy_lang("en_core_web_sm", disable=('entity_linker', 'entity_ruler', 'textcat',
                                          'textcat_multilabel', 'trainable_lemmatizer', 'morphologizer',
                                          'transformer', 'attribute_ruler', 'lemmatizer', 'ner'))
        # pretrained model from sBERT transformers: 'all-MiniLM-L6-v2'
        # self.model = SentenceTransformer('all-MiniLM-L6-v2')
        # pretrained model do not work so well on our dbpedia abstracts
        # fine-tuned with over 300k examples of sentence pairs: "/home/angelika/Phd/eTardis/Code/semanticDistance/fine_tuned_model_all-MiniLM-L6-v2"
        self.model = SentenceTransformer(sentence_embedding_model)
        # self.logger = logging.getLogger(__name__)

    def sentence_splitting_and_embed_sentences(self, text) -> list:
        doc = textacy.make_spacy_doc(text, lang=self.en)

        sentences = [sent.text for sent in doc.sents if sent.text.strip() != ""]
        embeddings = self.model.encode(sentences, batch_size=64, convert_to_numpy=True)
        embedding_dict = list(zip(sentences, embeddings))

        return embedding_dict


class SemanticDistance:
    def __init__(self, preprocessing):

        self.preprocessing = preprocessing
        self.preprocessing_dict = {}
        self.logger = logging.getLogger(__name__)

    def get_preprocessing_class(self):
        return self.preprocessing

    def reset_dictionaries(self) -> None:
        self.preprocessing_dict = {}
        return

    def nbow(self, document: list, vocab_len: int, dictionary) -> np.array:
        d = np.zeros(vocab_len, dtype=np.double)
        sentence_weight = 1/len(document)
        for idx in set(dictionary.doc2idx(document)):
            d[idx] = sentence_weight
        return d

    def sentmoversdistance(self, document1: List[tuple], document2: List[tuple]) -> float:
        """
        Compute Sentence Mover's distance among the two dicts of documents
        :param document1: list of tuples (sentence,embedding)
        :param document2: list of tuples (sentence,embedding)
        :return: smd as float
        """

        # print(document1, document2)
        is_doc_1_empty = len(document1) == 0
        is_doc_2_empty = len(document2) == 0

        if is_doc_1_empty and is_doc_2_empty:
            return 0.0
        elif is_doc_1_empty != is_doc_2_empty:
            print('At least one of the documents had no words that were'
                  'in the vocabulary. Aborting (returning inf).')
            return np.inf

        def __get_joined_tokens_for_all_sentences(sentence_tupels: list):
            return [sent[0] for sent in sentence_tupels]

        def __get_sent_embedding_dict(sentence_tupels: list):
           sent_embed_dict = {}
           for sent in sentence_tupels:
               sent_joined = sent[0]
               if not sent_joined in sent_embed_dict:
                   sent_embed_dict[sent_joined] = sent[1]
           return sent_embed_dict

        # somehow the shorter function does not work...
        # def __get_sent_embedding_dict(sentence_tupels: list):
        #    return dict(set(sentence_tupels))

        joined_tokens1 = __get_joined_tokens_for_all_sentences(document1)
        joined_tokens2 = __get_joined_tokens_for_all_sentences(document2)
        sent_embed_dict = __get_sent_embedding_dict(document1)
        sent_embed_dict.update(__get_sent_embedding_dict(document2))

        # print(sent_embed_dict)

        dictionary = Dictionary(documents=[joined_tokens1, joined_tokens2])
        vocab_len = len(dictionary)

        # Sets for faster look-up.
        #docset1 = set(joined_tokens1)
        #docset2 = set(joined_tokens2)
        distance_matrix = np.zeros((vocab_len, vocab_len), dtype=np.double)
        document2idx = set(dictionary.doc2idx(joined_tokens2))
        for i in set(dictionary.doc2idx(joined_tokens1)):
            sentence_vec1 = sent_embed_dict[dictionary[i]]
            for j in document2idx:
                # If the current cell is empty compute cosine distance between sentence vectors.
                #if not distance_matrix[i, j]:
                sentence_vec2 = sent_embed_dict[dictionary[j]]
                val = cosine(sentence_vec1, sentence_vec2)

                distance_matrix[i, j] = val
                # Fill the specular cell for saving computation
                distance_matrix[j, i] = val

        if not np.any(distance_matrix):
            # `emd` gets stuck if the distance matrix contains only zeros.
            print('The distance matrix is all zeros. Aborting (returning 0.0).')
            return 0.0

        # Compute nBOW representation of documents.
        d1 = self.nbow(joined_tokens1, vocab_len, dictionary)
        d2 = self.nbow(joined_tokens2, vocab_len, dictionary)

        smd = ot.emd2(d1, d2, distance_matrix)
        return smd

    def _put_text_in_prep_dict(self, text: str) -> None:
        if text not in self.preprocessing_dict:
            self.preprocessing_dict[text] = self.preprocessing.sentence_splitting_and_embed_sentences(text)
        return

    def sentence_movers_distance(self, text1: str, text2: str) -> dict:
        # similar to word movers distance only takes smallest distance between sentence vectors instead of word vectors

        self._put_text_in_prep_dict(text1)
        self._put_text_in_prep_dict(text2)
        text1_sentences = self.preprocessing_dict[text1]
        text2_sentences = self.preprocessing_dict[text2]

        smd = self.sentmoversdistance(text1_sentences, text2_sentences)
        return float(smd)


class FunctionsForSemanticScoreRetrieval:

    def __init__(self, semantic_tools):

        self.semantic_tools = semantic_tools
        self.preprocessing = self.semantic_tools.get_preprocessing_class()
        # self.logger = logging.getLogger(__name__)

    def get_semantic_score_for_two_texts(self, text1: str, text2: str,
                                         transform_to_similarity = False) -> float:
        """
                Compute distance for two texts.
                :param text1: string
                :param text2: string
                :return: distance score
        """
        semantic_score = self.semantic_tools.sentence_movers_distance(text1, text2)
        return np.exp(-semantic_score) if transform_to_similarity else semantic_score

    def get_semantic_score_for_reftext_and_listoftexts(self, ref_text: str, list_of_texts: list,
                                                       transform_to_similarity = False) -> np.ndarray:
        self.semantic_tools.reset_dictionaries()

        distance_matrix = np.zeros(len(list_of_texts), dtype=np.double)
        print(len(list_of_texts))

        for idx in range(len(list_of_texts)):
            if not distance_matrix[idx]:
                distance_matrix[idx] = self.get_semantic_score_for_two_texts(ref_text, list_of_texts[idx],
                                                                             transform_to_similarity)

        return distance_matrix.tolist()

    def get_semantic_score_for_all_combination_of_texts(self, list_of_texts: list,
                                                        transform_to_similarity = False) -> list:

        self.semantic_tools.reset_dictionaries()

        lenghtOfList = len(list_of_texts)
        distance_matrix = np.zeros((lenghtOfList, lenghtOfList), dtype=np.double)

        for idx1, text1 in enumerate(list_of_texts):
            idxStart = idx1 + 1  # MEL: brauche hier die +1, da alles andere nur die Diagonale befüllt
            for idx2, text2 in enumerate(list_of_texts[idxStart:], start=idxStart):

                if text1 != text2:
                    dscore = self.get_semantic_score_for_two_texts(text1, text2,
                                                                   transform_to_similarity)

                    distance_matrix[idx1][idx2] = dscore
                    distance_matrix[idx2][idx1] = dscore

        return distance_matrix.tolist()


# some classes to model country polygons, points in a geometry and
# find the continent, subregion and country names for a coordinate
class Point(object):
    """ Wrapper for ogr point """

    def __init__(self, lat, lng):
        """ Coordinates are in degrees """
        self.point = ogr.Geometry(ogr.wkbPoint)
        self.point.AddPoint(lng, lat)

    def getOgr(self):
        return self.point

    ogr = property(getOgr)


class Country(object):
    """ Wrapper for ogr country shape. Not meant to be instantiated directly. """

    def __init__(self, shape):
        self.shape = shape

    def getIso(self):
        return self.shape.GetField('ISO2')

    iso = property(getIso)

    def __str__(self):
        return self.shape.GetField('NAME')

    def contains(self, point):
        return self.shape.geometry().Contains(point)


class CountryChecker():
    """ Loads a country shape file, checks coordinates for country location. """

    def __init__(self):
        self.driver = ogr.GetDriverByName('ESRI Shapefile')
        self.worldMapFile = "./geo_data/TM_WORLD_BORDERS-0.3/TM_WORLD_BORDERS-0.3.shp"
        self.regionCodesFile = "./geo_data/region_codes.tsv"
        self.logger = logging.getLogger(__name__)

    def get_region_codes(self):
        region_codes = codecs.open(self.regionCodesFile, "r", "utf8")
        region_codes_dict = {}
        lines = region_codes.readlines()
        region_codes.close()
        for line in lines:
            line_list = line.strip().split("\t")
            if int(line_list[1].strip()):
                region_codes_dict[int(line_list[1].strip())] = str(line_list[0].strip().lower())
            else:
                region_codes_dict[line_list[1].strip()] = str(line_list[0].strip().lower())
        region_codes_dict[0] = "world"
        region_codes_dict[10] = "antarctica"
        return region_codes_dict

    def get_country_for_coordinate(self, lat, lon):
        """
        Checks given gps-incoming coordinates (i.e. latitude, longitude) for country.
        Output is either names for continent, subregion, country or None
        """
        region_codes_dict = self.get_region_codes()
        worldfile = self.driver.Open(self.worldMapFile)
        world_layer = worldfile.GetLayer()

        for i in range(world_layer.GetFeatureCount()):
            country = world_layer.GetFeature(i)

            if Country(country).contains(Point(lat, lon).ogr):

                continent = country.GetField("REGION")
                sub_region = country.GetField("SUBREGION")
                # Taiwan is somehow not provided as part of Asia (or any continent), so it is manually added here
                if country.GetField("NAME") == "Taiwan":
                    continent = 142
                    sub_region = 35
                if country.GetField("NAME") == "Antarctica":
                    continent = 10
                    sub_region = 10

                # self.logger.info(region_codes_dict[continent] + "," + region_codes_dict[sub_region] +
                #                 "," + country.GetField("Name"))
                return region_codes_dict[continent], region_codes_dict[sub_region], Country(country).__str__()

        #Nothing Found
        return "", "", ""

    def get_continent_countries_dict(self):

        """ returns a dictionary of all available continents: subregions: list of countries """

        self.logger.info("Build dictionary with regional iso codes")
        region_codes_dict = self.get_region_codes()
        self.logger.info(str(region_codes_dict))
        continent_countries = {}
        worldFile = self.driver.Open(self.worldMapFile)
        world_layer = worldFile.GetLayer()

        for i in range(world_layer.GetFeatureCount()):
            region = world_layer.GetFeature(i)

            continent = region.GetField("REGION")
            sub_region = region.GetField("SUBREGION")
            #Taiwan is somehow not provided as part of Asia (or any continent), so it is manually added here
            if region.GetField("NAME") == "Taiwan":
                continent = 142
                sub_region = 35
            if region.GetField("NAME") == "Antarctica":
                continent = 10
                sub_region = 10
            if region_codes_dict[continent] not in continent_countries:
                continent_countries[region_codes_dict[continent]] = {}
            if region_codes_dict[sub_region] not in continent_countries[region_codes_dict[continent]]:
                continent_countries[region_codes_dict[continent]][region_codes_dict[sub_region]] = []

            # self.logger.info(region_codes_dict[continent]+","+region_codes_dict[sub_region]+","+region.GetField("Name"))
            continent_countries[region_codes_dict[continent]][region_codes_dict[sub_region]].append(region.GetField("NAME"))
        return continent_countries


class GraphConnectionManager:
    def __init__(self, url: str, username: str, password: str):
        """
        Initialise Connection manager to handle access to neo4j graph instance.
        :param url: url to access neo4j graph instance
        :param username: username to access neo4j graph instance
        :param password: password to access neo4j graph instance
        """
        self.url = url
        self.username = username
        self.password = password
        self.connection = None
        self.logger = logging.getLogger(__name__)

    def __enter__(self):
        """
        This function is called whenever GraphConnectionManager instance is called.
        :return:
        """
        try:
            self.connection = Graph(self.url, auth=(self.username, self.password))
        except Exception as e:
            self.logger.error("Could connect to neo4j database.")
            raise RuntimeError(e)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        This function is called after GraphConnectionManager instance is called.
        :param exc_type:
        :param exc_val:
        :param exc_tb: indicates error during transaction
        :return:
        """
        # TODO: add handling of commit and rollback for both cases
        # if exc_tb is None:
        #     self.logger.info("Executed transaction.")
        # else:
        #     self.logger.error("Could not execute transaction. Rollback happened.")
        pass


class GraphApi:

    def __init__(self, url, username, password):
        """
        Initialise Graph api which is used to handle communication between neo4j graph database and any services.
        :param url: url to access neo4j graph instance
        :param username: username to access neo4j graph instance
        :param password: password to access neo4j graph instance
        """
        self.placeholder = "unknown"
        self.toleranceGeoLocations_default = 0.01
        self.default_include_unknown = 1
        self.url = url
        self.username = username
        self.password = password
        self.logger = logging.getLogger(__name__)
        self.country_checker = CountryChecker()
        self.semantic_functions = self.get_semantic_functions()
        self.uris_image_urls = self._ld_pkl("../../data/uris_images.pkl")

    def get_semantic_functions(self):
        preprocessing = TextPreprocessing(sentence_embedding_model="./fine_tuned_model_all-MiniLM-L6-v2")
        semantic_tools = SemanticDistance(preprocessing=preprocessing)
        return FunctionsForSemanticScoreRetrieval(semantic_tools=semantic_tools)

    def _dump_data(self, data: Dict, data_path: str, file_name: str) -> None:
        """
        Helper function to dump any json formatted data into a specified file.
        :param data: dict is represented in json format
        :param data_path: path where file is stored
        :param file_name: name of created file
        :return:
        """
        with open(os.path.join(data_path, file_name), 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    def _ld_pkl(self, fname:str) -> dict:
        with open(fname, 'rb') as f:
            # The protocol version used is detected automatically, so we do not
            # have to specify it.
            data = pickle.load(f)
            return data

    def _format_uri(self, uri: str) -> str:
        """
        Helper function to reformat representation of dbpedia uris.
        :param uri: dbpedia resource identifier
        :return: string contains reformatted uri representation
        """
        # TODO: same as in graph_handler
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
        elif "http://www.w3.org/2004/02/skos/core#" in uri:
            uri = uri.replace("http://www.w3.org/2004/02/skos/core#", "w3:")
        elif "http://purl.org/dc/terms/" in uri:
            uri = uri.replace("http://purl.org/dc/terms/", "purl:")
        elif "http://purl.org/dc/elements/1.1/" in uri:
            uri = uri.replace("http://purl.org/dc/elements/1.1/", "purl:")
        elif "http://purl.org/linguistics/gold/" in uri:
            uri = uri.replace("http://purl.org/linguistics/gold/", "purl:")
        elif "http://xmlns.com/foaf/0.1/" in uri:
            uri = uri.replace("http://xmlns.com/foaf/0.1/", "xmlns:")
        return uri.strip()

    def _reformat_locations(self, locations: str) -> List:
        """
        Helper function to create proper location representation as tuple (latitude, longitude)
        :param locations: string representation of list of coordinates
        :return: List contains formatted representations of coordinates
        """
        try:
            if self.placeholder in locations:
                return locations
            else:
                # returned format is (lat, long)
                return [(float(loc[1]), float(loc[0])) for loc in ast.literal_eval(locations)]
        except (IndexError, ValueError):
            self.logger.warning("Could not handle location formatting %s.", locations)
            return locations

    def _deg2rad(self, degree:float) -> float:
        """
        Helper function to get a degree as radian value
        param degree: Float for Longitude or Latitude
        returns: radian value
        """
        radian = degree * (math.pi / 180)
        return radian

    def _hav(self, radian):
        hav = math.sin(radian / 2) * math.sin(radian / 2)
        return hav

    def _getDistance(self, latitude1, longitude1, latitude2, longitude2) -> float:
        """
        Helper function to get the distance in km between two coordinates given as latitude and longitude
        makes use of the Haversine Formula (great-circle distance between two points on a sphere
                                            given their longitudes and latitudes)
        van Brummelen, Glen Robert (2013). Heavenly Mathematics: The Forgotten Art of Spherical Trigonometry.
        Princeton University Press. ISBN 9780691148922. 0691148929. Retrieved 2015-11-10.

        de Mendoza y Ríos, Joseph (1795). Memoria sobre algunos métodos nuevos
        de calcular la longitud por las distancias lunares:
        y aplicacion de su teórica á la solucion de otros problemas de navegacion (in Spanish). Madrid, Spain: Imprenta Real.

        """
        earth_radius = 6371

        dLat = self._deg2rad(latitude2 - latitude1)
        dLon = self._deg2rad(longitude2 - longitude1)

        a = self._hav(dLat) + (1 - self._hav(self._deg2rad(latitude1 - latitude2)) - \
                               self._hav(self._deg2rad(latitude1 + latitude2))) * self._hav(dLon)
        #formula with cosine below (but the use of cosine causes some mathematical problems in small distances)
        #a = math.sin(dLat / 2) * math.sin(dLat / 2) + math.cos(self._deg2rad(latitude1)) * math.cos(
        #    self._deg2rad(latitude2)) * math.sin(dLon / 2) * math.sin(dLon / 2)
        # a = sin(dlat/2)^2 + cos(rad(lat1)) * cos(rad(lat2)) * sin(dLon/2)^2
        #sin = the sine of x radians, cos = the cosine of x radians

        c = 2 * math.asin(math.sqrt(a))
        # asin = inverse hyperbolic sine of sqrt(a)
        d = earth_radius * c
        # convert to km

        return d

    def _CompareLonOrLat(self, coords_set:set, idx:int, coord_part_comp:float,
                         dist:float, neg_dist:float, results=set()) -> set:

        """
        Helper function to to test if coordinates are within a tolerance distance for latitude or longitude
        param coords_set: Set of retrieved coordinates from database
        param idx: index as integer 0 or 1 to choose longitude (idx:1) or latitude (idx:0) as reference
        param coord_part_comp: Float for coordinate Longitude or Latitude
        param dist: Float for tolerance to the positive side
        param neg_dist: Float for tolerance to the negative side
        param results: Set of coordinates in range
        return: Set of resulting coordinates
        """

        if coord_part_comp < 0:
            for coords in coords_set:
                try:
                    if coords[idx] >= - dist and coords[idx] <= -neg_dist:
                        results.add(coords)
                except:
                    continue
        else:
            for coords in coords_set:
                try:
                    if coords[idx] <= dist and coords[idx] >= neg_dist:
                        results.add(coords)
                except:
                    continue
        return results

    def _test_for_near_locations_byCoords(self, latitude = None, longitude = None, dist = 0.01) -> set:
        """
        Helper function to retrieve a list of all coordinates in a certain radius from the database
        param latitude: Float or None for latitude
        param longitude: Float or None for longitude
        param dist: float for tolerance (if one coord is missing) or search radius
        return: Set of relevant coordinates from database
        """
        all_coords = self.retrieve_all_coordinates()
        all_coords_set = set()
        for locations in all_coords:
            try:
                locations_formatted = self._reformat_locations(locations)
                for location in locations_formatted:
                    all_coords_set.add(location)
            except:
                continue

        # self.logger.info(str(longitude)+", "+str(latitude)+"; "+str(dist))

        results = set()
        if longitude and latitude:
            for coords in all_coords_set:
                try:
                    distance = self._getDistance(latitude, longitude, coords[0], coords[1])
                    if distance < dist:
                        results.add(coords)
                except:
                    continue
        elif not latitude and longitude:
            # self.logger.info("Only Longitude available")
            idx = 1
            neg_dist = abs(longitude)-dist
            dist += abs(longitude)
            # self.logger.info(str(longitude) + ", " + str(latitude) + "; " + str(dist)+ ", "+str(neg_dist))
            results = self._CompareLonOrLat(all_coords_set, idx, longitude, dist, neg_dist, results)
        elif not longitude and latitude:
            #self.logger.info("Only Latitude available")
            idx = 0
            neg_dist = abs(latitude) - dist
            dist += abs(latitude)
            # self.logger.info(str(longitude) + ", " + str(latitude) + "; " + str(dist)+ ", "+str(neg_dist))
            results = self._CompareLonOrLat(all_coords_set, idx, latitude, dist, neg_dist, results)

        return results

    def _list_is_empty(self, list_to_test):

        if list_to_test == []:
            return True
        else:
            return False

    def _reformat_relation_properties(self, properties: Dict) -> Dict:
        """
        Helper function to format names of properties from dbpedia into short representation
        :param properties: dict of properties consists of names as keys and corresponding values
        :return: dict contains formatted propertiy representations
        """
        formatted_props = {}
        for k, v in properties.items():
            formatted_props[k] = self._format_uri(v)
        return formatted_props

    def _map_connection_properties(self, label: str, properties: OrderedDict) -> OrderedDict:
        """
                Helper function to convert the properties of a neo4j node into hierarchical description + x format
                :param label: string specifies the type of the node
                :param properties: dict contains properties as key, value pairs
                :return: OrderedDict represents reformatted node format
                """
        node = OrderedDict()
        node["name"] = properties.pop("name") if "name" in properties else "unknown"
        node["subcategories"] = properties.pop("subclasses") if "subclasses" in properties else "unknown"
        node["period"] = properties.pop("period") if "period" in properties else "unknown"
        node["locations"] = self._reformat_locations(
            properties.pop("locations")) if "locations" in properties else "unknown"
        # type specific descriptions
        if label == "Event":
            node["facets"] = properties.pop("facets") if "facets" in properties else "unknown"
            node["participants"] = properties.pop("participants") if "participants" in properties else "unknown"
        elif label == "Agent":
            node["genders"] = properties.pop("genders") if "genders" in properties else "unknown"
            node["families"] = properties.pop("families") if "families" in properties else "unknown"
            node["positions"] = properties.pop("positions") if "positions" in properties else "unknown"
            node["religions"] = properties.pop("religions") if "religions" in properties else "unknown"
        elif label == "TopicalConcept":
            node["facets"] = properties.pop("facets") if "facets" in properties else "unknown"
            node["uses"] = properties.pop("uses") if "uses" in properties else "unknown"
        elif label == "CulturalArtifact":
            node["authors"] = properties.pop("authors") if "authors" in properties else "unknown"
        elif label == "MaterialObject":
            node["material"] = properties.pop("material") if "material" in properties else "unknown"
        elif label == "Place":
            pass
        elif label == "TimePeriod":
            pass
        elif label == "Miscellaneous":
            pass
        # store further properties for extended view
        node["properties"] = properties
        return node


    def _map_properties(self, label: str, properties: Dict) -> OrderedDict:
        """
        Helper function to convert the properties of a neo4j node into hierarchical description + x format
        :param label: string specifies the type of the node
        :param properties: dict contains properties as key, value pairs
        :return: OrderedDict represents reformatted node format
        """
        node = OrderedDict()
        node["name"] = properties.pop("name") if "name" in properties else "unknown"
        node["subcategories"] = properties.pop("subclasses") if "subclasses" in properties else "unknown"
        node["period"] = properties.pop("period") if "period" in properties else "unknown"
        node["locations"] = self._reformat_locations(
            properties.pop("locations")) if "locations" in properties else "unknown"
        # type specific descriptions
        if label == "Event":
            node["facets"] = properties.pop("facets") if "facets" in properties else "unknown"
            node["participants"] = properties.pop("participants") if "participants" in properties else "unknown"
        elif label == "Agent":
            node["genders"] = properties.pop("genders") if "genders" in properties else "unknown"
            node["families"] = properties.pop("families") if "families" in properties else "unknown"
            node["positions"] = properties.pop("positions") if "positions" in properties else "unknown"
            node["religions"] = properties.pop("religions") if "religions" in properties else "unknown"
        elif label == "TopicalConcept":
            node["facets"] = properties.pop("facets") if "facets" in properties else "unknown"
            node["uses"] = properties.pop("uses") if "uses" in properties else "unknown"
        elif label == "CulturalArtifact":
            node["authors"] = properties.pop("authors") if "authors" in properties else "unknown"
        elif label == "MaterialObject":
            node["material"] = properties.pop("material") if "material" in properties else "unknown"
        elif label == "Place":
            pass
        elif label == "TimePeriod":
            pass
        elif label == "Miscellaneous":
            pass
        # store further properties for extended view
        node["properties"] = properties
        return node

    def _create_relationship_object(self, source: str, target: Dict, name: str, properties: Dict,
                                    weights: Dict) -> OrderedDict:
        """
        Helper function to create a dict object of a relation between two nodes.
        :param source: string specifies source node
        :param target: string specifies target node
        :param name: string specifies relation name which depends on target node
        :param properties: list specifies properties of relation
        :param weights: dict contains weights of relations
        :return:
        """
        relation = OrderedDict()
        relation["source"] = source
        relation["name"] = name
        relation["properties"] = properties
        relation["weights"] = weights
        relation["target"] = target
        return relation

    def _create_label_query_param(self, labels: List, node_mode: bool) -> str:
        """
        Helper function to create different cypher query representations when searching for labels
        :param labels: list contains labels
        :param node_mode: adjust query representation for different use cases
        :return: str contains cypher query part
        """
        available_labels = self.retrieve_available_labels()
        # check if all labels are available in graph database
        for lbl in labels:
            if lbl not in available_labels:
                self.logger.error("Could not handle unknown label %s.", lbl)
                raise Exception(f"Input contains unknown label: {lbl}")
        if node_mode:
            query_params = " OR ".join(["n:" + lbl for lbl in labels])
        else:
            query_params = f"type(r) IN {['has' + str(lbl).capitalize() for lbl in labels]}"
        if query_params:
            query_params = f" WHERE ({query_params}) "
        return query_params

    def evaluate_as_bool(self, int_val:int) -> bool:

        """
        Helper function to convert int 0 or 1 to bool
        :param int_val: int 0 or 1
        :return: bool or None
        """

        if int_val == 1:
            bool_val = True
        elif int_val == 0:
            bool_val = False
        else:
            bool_val = None

        return bool_val

    def retrieve_statistics(self) -> Tuple:
        """
        Retrieve the number of available nodes and relations in corresponding neo4j graph database
        :return: tuple contains two integer values
        """
        with GraphConnectionManager(self.url, self.username, self.password) as graph:
            query = """
                MATCH (n)
                RETURN COUNT(n) AS count;
                """
            number_nodes = graph.connection.run(query).evaluate()
            query = """
                MATCH ()-[r]->()
                RETURN COUNT(r) AS count;
                """
            number_relations = graph.connection.run(query).evaluate()
            return number_nodes, number_relations

    def retrieve_available_uris(self, labels = []) -> List:

        """
        Retrieve all unique uris in the database
        :param labels: list of labels the uris are filtered with
        : return: List of Strings
        """
        if self._list_is_empty(labels):
            labels = ["Agent", "CulturalArtifact", "Event", "MaterialObject", "Miscellaneous", "Place",
                      "TimePeriod", "TopicalConcept"]
        with GraphConnectionManager(self.url, self.username, self.password) as graph:
            all_available_uris = set()
            for label in labels:
                query = f"""
                        MATCH (n:{label}) RETURN n.uri
                        """
                results = graph.connection.run(query)
                for res in results:
                    all_available_uris.add(res.get("n.uri"))

            return list(all_available_uris)

    def retrieve_timespan(self) -> List:
        """
        Retrieve the year for the (annotated) earliest and (annotated) latest entry in the database.
        :return: list with two integers
        """
        with GraphConnectionManager(self.url, self.username, self.password) as graph:
            query = f"""
                        MATCH (n)-[r]->(m) 
                        RETURN m.period AS period
                        """
            results = graph.connection.run(query)
            min_time_point = 0
            max_time_point = 0
            for res in results:
                if res.get("period") == "unknown":
                    continue
                start_time = int(res.get("period")[0])
                end_time = int(res.get("period")[1])
                if min_time_point == 0:
                    min_time_point = start_time
                    max_time_point = end_time

                if start_time < min_time_point:
                    min_time_point = start_time
                if end_time > max_time_point:
                    max_time_point = end_time
            return [min_time_point, max_time_point]

    def retrieve_all_coordinates(self) -> List:

        """
        Retrieve a set of all available coordinates in the database
        : return: List of tuples with two floats for longitude and latitude
        """

        with GraphConnectionManager(self.url, self.username, self.password) as graph:
            available_cooordinates = set()
            query = f"""
                        MATCH (n)-[r]->(m) 
                        RETURN m.locations AS locations
                        """
            results = graph.connection.run(query)
            for res in results:
                try:
                    locations = res.get("locations")
                    available_cooordinates.add(locations)
                except:
                    continue
            return list(available_cooordinates)

    def retrieve_available_subcategories(self) -> List:
        """
        Retrieve all available subcategories (types of nodes)
                 which consist of dbpedia classes.
        :return: list contains subcategories as string values
        """
        with GraphConnectionManager(self.url, self.username, self.password) as graph:
            available_subcategories = set()
            query = f"""
                        MATCH (n)-[r]->(m) 
                        RETURN m.subclasses AS subclasses, m.period AS period, m.locations AS locations
                        """
            results = graph.connection.run(query)
            for res in results:
                subcats = res.get("subclasses")
                for subcat in subcats:
                    available_subcategories.add(subcat)
            return list(available_subcategories)

    def retrieve_available_labels(self) -> List:
        """
        Retrieve all available labels / toplevel categories (types of nodes)
                 which consist of reduced number of dbpedia super classes.
        :return: list contains labels as string values
        """
        with GraphConnectionManager(self.url, self.username, self.password) as graph:
            query = "CALL db.labels()"
            results = graph.connection.run(query)
            available_labels = sorted([str(rec.get("label")) for rec in results])
            return available_labels

    def retrieve_available_properties(self) -> List:
        """
        Retrieve list of all available property identifiers over all nodes.
        :return: list consists of dicts which contain uri and formatted uri
                 for each distinct property identifier
        """
        with GraphConnectionManager(self.url, self.username, self.password) as graph:
            query = """
                MATCH (n)-[r]-(m) 
                UNWIND properties(r) AS props
                RETURN apoc.coll.flatten(COLLECT(DISTINCT apoc.map.values(props, keys(props))))
                """
            available_properties = graph.connection.run(query).evaluate()
            return sorted([{"uri": prop, "formattedUri": self._format_uri(prop)} for prop in available_properties],
                          key=lambda x: x["formattedUri"])

    def retrieve_available_countries(self, continents:List) -> List:

        """
        Retrieve list of available countries
        :param continents: List of strings for filter by specified continents
        :return: List of strings
        """

        continent_dict = self.country_checker.get_continent_countries_dict()

        countries = []
        if not self._list_is_empty(continents):
            keys = []
            for key in continents:
                keys.append(key.strip().lower())
        else:
            keys = list(continent_dict.keys())

        for key in keys:
            for sub_region in continent_dict[key]:
                countries += continent_dict[key][sub_region]

        countries = sorted(list(set(countries)))
        self.logger.info("%s Countries for specified Continents from dict")
        del continent_dict

        return countries

    def retrieve_available_continents(self) -> List:

        continent_subregions_countries = self.country_checker.get_continent_countries_dict()
        continents = []
        for continent in continent_subregions_countries:
            continents.append(continent)
        del continent_subregions_countries
        return sorted(continents)

    def retrieve_available_sub_regions(self) -> List:

        continent_subregions_countries = self.country_checker.get_continent_countries_dict()
        sub_regions = []
        for continent in continent_subregions_countries:
            for sub_region in continent_subregions_countries[continent]:
                sub_regions.append(sub_region)
        del continent_subregions_countries
        return sorted(sub_regions)

    def _filter_is_contemporary(self, is_contemporary:int, sub_categories:List) -> bool:

        """
        Helper function to filter with param is_contemporary
        :param is_contemporary: None, 0, 1
        :param sub_categories: List of strings
        :return: bool if node is included
        """

        include_node = True

        if is_contemporary == 0:
            # self.logger.info("%s exclude Persons")
            if "dbo:Person" in sub_categories:
                include_node = False
        elif is_contemporary == 1:
            if not "dbo:Person" in sub_categories:
                # self.logger.info("%s include only Persons")
                include_node = False
        else:
            # self.logger.info("%s include Persons and non-Persons")
            include_node = True

        return include_node

    def _filter_time_information(self, time_info, filteredStartyear:int, filteredEndyear:int,
                                  takeAsPeriod:int, include_unknown_period:int) -> bool:

        include_node = False

        if "unknown" in time_info:
            include_node = self._test_for_unkown(include_unknown_period)
        else:
            if not filteredStartyear and not filteredEndyear:
                return True

            if takeAsPeriod == 1:
                if filteredStartyear and filteredEndyear:
                    if time_info[0] >= filteredStartyear and time_info[1] <= filteredEndyear:
                        include_node = True
                elif filteredStartyear and not filteredEndyear:
                    if time_info[0] >= filteredStartyear:
                        include_node = True
                elif not filteredStartyear and filteredEndyear:
                    if time_info[1] <= filteredEndyear:
                        include_node = True
                else:
                    include_node = False
            elif takeAsPeriod == 0:
                if filteredStartyear and filteredEndyear:
                    if time_info[0] == filteredStartyear and time_info[1] == filteredEndyear:
                        include_node = True
                elif filteredStartyear and not filteredEndyear:
                    if time_info[0] == filteredStartyear and time_info[1] == filteredStartyear:
                        include_node = True
                elif not filteredStartyear and filteredEndyear:
                    if time_info[1] == filteredEndyear and time_info[1] == filteredEndyear:
                        include_node = True
                else:
                    include_node = False
            else:
                #self.logger.info("%s ignore time information")
                include_node = True

        return include_node

    def _test_if_coords_are_in_search_space(self, coords:list, near_coordinates:set, include_node=False) -> bool:

        for coord in coords:
            try:
                if coord in near_coordinates:
                    include_node = True
                    break
            except:
                continue

        return include_node

    def _test_if_coords_are_in_continent_country(self, coords, search_regions, include_node=False) -> bool:

        if not search_regions:
            include_node = True
            self.logger.info("No terms in continents and countries."+" "+str(include_node))
            return include_node

        for coord in coords:
            try:
                continent, subregion, country = self.country_checker.get_country_for_coordinate(coord[0], coord[1])
                if continent in search_regions or subregion in search_regions or country.lower() in search_regions:
                    include_node = True
                    break
            except:
                continue
        return include_node

    def _test_for_unkown(self, include_unknown:int) -> bool:

        if include_unknown == 1:
            include_node = True
        elif include_unknown == 0:
            include_node = False
        else:
            include_node = False

        return include_node

    def _filter_location(self, locations:list, near_coordinates:list) -> bool:
        coords = self._reformat_locations(locations)
        include_node = self._test_if_coords_are_in_search_space(coords, near_coordinates)
        return include_node

    def _get_location_Continent_Country_ByCoords(self, locations:list) -> Dict:

        location_dict_list = []
        if "unknown" in locations:
            return location_dict_list

        if not type(locations) is list:
            coords = self._reformat_locations(locations)
        else:
            coords = locations

        for coord in coords:
            try:
                continent, subregion, country = self.country_checker.get_country_for_coordinate(coord[0], coord[1])
                location_dict_list.append({"coordinates": list(coord),
                                           "continent": continent,
                                           "sub_region": subregion,
                                           "country": country})
            except Exception as e:
                self.logger.critical(e)
                continue

        return location_dict_list

    def _filter_location_ByName(self, locations:list, search_regions:list,
                                include_unknown_location:int) -> bool:

        if "unknown" in locations:
            include_node = self._test_for_unkown(include_unknown_location)
        else:
            coords = self._reformat_locations(locations)
            include_node = self._test_if_coords_are_in_continent_country(coords, search_regions)

        return include_node

    def _get_image_urls_for_node(self, node):

        if '"'+node["uri"]+'"' in self.uris_image_urls:
            node["imageUrls"] = self.uris_image_urls['"'+node["uri"]+'"']
        elif node["uri"] in self.uris_image_urls:
            node["imageUrls"] = self.uris_image_urls[node["uri"]]
        else:
            self.logger.info("uri "+node["uri"] +" is not in image database")
            node["imageUrls"] = []

        return node

    def _returnline_for_counter(self):
        return f"""RETURN DISTINCT COUNT(n) AS count"""

    def _returnline_for_counterConnections(self, threshold:int) -> str:
        threshold_query_param = self._create_threshold_query_param(threshold)
        return f"""RETURN DISTINCT COUNT(m) AS count {threshold_query_param} """

    def _returnline_for_node_retrieval(self, location_param=f""):

        return_line = f"""RETURN n{location_param}
                          ORDER BY n.uri
                       """

        return return_line

    def _returnline_for_more_extensive_query(self):

        return_line2 = f"ORDER BY n.uri"
        return_line = f"""RETURN DISTINCT labels(n), n.uri, n.name, n.subclasses, n.period, n.locations
                          {return_line2}
                       """

        return return_line

    def _returnline_for_more_extensive_queryConnections(self, threshold:int, location_param:str) -> str:

        threshold_query_param = self._create_threshold_query_param(threshold)
        return_line2 = f" ORDER BY m.uri "
        return_line = f"""RETURN DISTINCT r, m{location_param} {return_line2} {threshold_query_param}"""

        return return_line

    def _returnline_for_GeoFilter(self, is_contemporary:int, include_unknown_period:int, takeAsPeriod:int,
                                  filteredStartyear:int, filteredEndyear:int, latitude:float, longitude:float,
                                  include_unknown_location:int, node_mode=True, threshold=10,
                                  location_param=f", m.locations") -> str:

        if not is_contemporary and include_unknown_period == 1 and not takeAsPeriod \
                and not filteredStartyear and not filteredEndyear and not latitude \
                and not longitude and include_unknown_location == 1:
            if node_mode:
                return_line = self._returnline_for_counter()
            else:
                return_line = self._returnline_for_counterConnections(threshold, location_param)
        else:
            if node_mode:
                return_line = self._returnline_for_more_extensive_query()
            else:
                return_line = self._returnline_for_more_extensive_queryConnections(threshold, location_param)

        return return_line

    def _returnline_for_GeoFilterByName(self, is_contemporary:int, include_unknown_period:int, takeAsPeriod:int,
                                        filteredStartyear:int, filteredEndyear:int, continents:list, countries:list,
                                        include_unknown_location:int, node_mode=True, threshold=10,
                                        location_param=f", m.locations") -> str:

        if not is_contemporary and include_unknown_period == 1 and not takeAsPeriod \
                and not filteredStartyear and not filteredEndyear and not continents \
                and not countries and include_unknown_location == 1:
            if node_mode:
                return_line = self._returnline_for_counter()
            else:
                return_line = self._returnline_for_counterConnections(threshold)
        else:
            if node_mode:
                return_line = self._returnline_for_more_extensive_query()
            else:
                return_line = self._returnline_for_more_extensive_queryConnections(threshold, location_param)

        return return_line

    def _create_threshold_query_param(self, threshold:int) -> str:

        threshold_query_param = ""
        if threshold:
            threshold_query_param = f"LIMIT {int(threshold)}"

        return threshold_query_param

    def _returnline_for_connections(self, threshold:int, location_param=f"", n=10):

        if not threshold:
            threshold = 0
        threshold_query_param = self._create_threshold_query_param(threshold*n)
        return_line = f"RETURN r, m {location_param} {threshold_query_param}"

        return return_line

    def _build_query_connections(self, uri:str, return_line:str, user_input: str, search_in_all_properties: bool,
                                 filtered_labels: List, filtered_properties: List) -> str:

        if (not user_input or user_input.strip() == "") and not filtered_labels and not filtered_properties:

            #cypher: length (n-[r]->(m)) 0 if it is incoming to n and 1 if it is outgoing
            # -> shows direction of relation = outgoing
            query = f"""
                        MATCH (n {{ uri: "{uri}" }})-[r]->(m)
                        {return_line}
                    """
            # LIMIT 50 entfernen -> Melanie will alle nodes, zum debuggen limit drinlassen ^^
        else:
            label_filter = ""
            if filtered_labels:
                label_filter = self._create_label_query_param(filtered_labels, False)
            property_filter = ""
            if filtered_properties:
                property_filter = f"ANY(k IN keys(m) WHERE k in {filtered_properties})"
            if user_input:
                #28 is the length of uri preamble 'http://dbpedia.org/resource/'
                search_in_name_and_uri = f'OR (substring(toLower(toString(m.uri)), 28) CONTAINS "{user_input}" OR ANY(x IN m.name WHERE toLower(toString(x)) CONTAINS "{user_input}"))'
                if property_filter:
                    property_filter = "AND " + property_filter
                if search_in_all_properties:
                    query = f"""
                                MATCH (n {{ uri: "{uri}" }})-[r]->(m)
                                {label_filter}        
                                UNWIND properties(m) AS props
                                WITH apoc.coll.flatten(apoc.coll.flatten(COLLECT(DISTINCT apoc.map.values(props, keys(props))))) AS flattenProps, keys(props) AS k, r AS r, m AS m
                                WHERE (ANY(x IN flattenProps WHERE toLower(toString(x)) CONTAINS "{user_input}") OR size(flattenProps) is null)
                                {search_in_name_and_uri}
                                {property_filter}
                                {return_line}
                            """
                else:
                    query = f"""
                                MATCH (n {{ uri: "{uri}" }})-[r]->(m)
                                {label_filter}
                                UNWIND properties(m) AS props
                                WITH apoc.coll.flatten(apoc.coll.flatten(COLLECT(DISTINCT apoc.map.values(props, ['dbo:abstract'])))) AS flattenProps, keys(props) AS k, r AS r, m AS m
                                WHERE (ANY(x IN flattenProps WHERE toLower(toString(x)) CONTAINS "{user_input}") OR size(flattenProps) is null)
                                {search_in_name_and_uri}
                                {property_filter}
                                {return_line}
                            """
            else:
                if property_filter and not label_filter:
                    property_filter = "WHERE " + property_filter
                elif property_filter and label_filter:
                    property_filter = "AND " + property_filter
                query = f"""
                            MATCH (n {{ uri: "{uri}" }})-[r]->(m)
                            {label_filter}
                            {property_filter}
                            {return_line}
                        """
        return query

    def _build_query(self, return_line:str, user_input: str, search_in_all_properties: bool,
                     filtered_labels: List, filtered_properties: List) -> str:

        if not user_input and not filtered_labels and not filtered_properties:

            query = f"""
                        MATCH (n)
                        {return_line}
                    """
            # LIMIT 50 entfernen -> Melanie will alle nodes, zum debuggen limit drinlassen ^^
        else:
            label_filter = ""
            if filtered_labels:
                label_filter = self._create_label_query_param(filtered_labels, True)
            property_filter = ""
            if filtered_properties:
                property_filter = f"ANY(k IN keys(n) WHERE k in {filtered_properties})"
            if user_input:
                # 28 is the length of uri preamble 'http://dbpedia.org/resource/'
                search_in_name_and_uri = f'OR (substring(toLower(toString(n.uri)), 28) CONTAINS "{user_input}" OR ANY(x IN n.name WHERE toLower(toString(x)) CONTAINS "{user_input}"))'
                if property_filter:
                    property_filter = "AND " + property_filter
                if search_in_all_properties:
                    query = f"""
                                MATCH (n)
                                {label_filter}
                                UNWIND properties(n) AS props
                                WITH apoc.coll.flatten(apoc.coll.flatten(COLLECT(DISTINCT apoc.map.values(props, keys(props))))) AS flattenProps, keys(props) AS k, n AS n
                                WHERE (ANY(x IN flattenProps WHERE toLower(toString(x)) CONTAINS "{user_input}") OR size(flattenProps) is null)
                                {search_in_name_and_uri}
                                {property_filter}
                                {return_line}
                            """
                else:
                    query = f"""
                                MATCH (n)
                                {label_filter}
                                UNWIND properties(n) AS props
                                WITH apoc.coll.flatten(apoc.coll.flatten(COLLECT(DISTINCT apoc.map.values(props, ['dbo:abstract'])))) AS flattenProps, keys(props) AS k, n AS n
                                WHERE (ANY(x IN flattenProps WHERE toLower(toString(x)) CONTAINS "{user_input}") OR size(flattenProps) is null)
                                {search_in_name_and_uri}
                                {property_filter}
                                {return_line}
                            """
            else:
                if property_filter and not label_filter:
                    property_filter = "WHERE " + property_filter
                elif property_filter and label_filter:
                    property_filter = "AND " + property_filter
                query = f"""
                            MATCH (n)
                            {label_filter}
                            {property_filter}
                            {return_line}
                        """
        return query

    def _combine_return_line_and_query_GeoFilter(self, user_input:str, search_in_all_properties:str,
                                                 filtered_labels:list, filtered_properties:list, is_contemporary:int,
                                                 include_unknown_period:int, takeAsPeriod:int, filteredStartyear:int,
                                                 filteredEndyear:int, latitude:float, longitude:float,
                                                 include_unknown_location:int, node_mode=True, threshold=10, uri="",
                                                 location_param=f", m.locations") -> str:

        return_line = self._returnline_for_GeoFilter(is_contemporary, include_unknown_period, takeAsPeriod,
                                                     filteredStartyear, filteredEndyear, latitude, longitude,
                                                     include_unknown_location, node_mode, threshold, location_param)
        if node_mode:
            query = self._build_query(return_line, user_input, search_in_all_properties, filtered_labels,
                                      filtered_properties)
        else:
            query = self._build_query_connections(uri, return_line,user_input, search_in_all_properties,
                                                  filtered_labels, filtered_properties)

        return query

    def _combine_return_line_and_query_GeoFilterByName(self, user_input:str, search_in_all_properties:str,
                                                 filtered_labels:list, filtered_properties:list, is_contemporary:int,
                                                 include_unknown_period:int, takeAsPeriod:int, filteredStartyear:int,
                                                 filteredEndyear:int, continents:list, countries:list,
                                                 include_unknown_location:int, node_mode=True, threshold=10, uri="",
                                                 location_param=f", m.locations") -> str:

        return_line = self._returnline_for_GeoFilterByName(is_contemporary, include_unknown_period, takeAsPeriod,
                                                           filteredStartyear, filteredEndyear, continents,
                                                           countries, include_unknown_location, node_mode, threshold,
                                                           location_param)

        if node_mode:
            query = self._build_query(return_line, user_input, search_in_all_properties, filtered_labels,
                                      filtered_properties)
        else:
            query = self._build_query_connections(uri, return_line, user_input, search_in_all_properties,
                                                  filtered_labels, filtered_properties)

        return query

    def retrieve_counter_nodes_by_inputWithGeoFilter(self, user_input: str, search_in_all_properties: bool,
                                                     filtered_labels: List, is_contemporary: int,
                                                     filtered_properties: List, takeAsPeriod: int,
                                                     include_unknown_period: int, filteredStartyear: int,
                                                     filteredEndyear: int, include_unknown_location: int,
                                                     latitude:float, longitude:float, tolance_or_radius:float) -> int:
        """
        Retrieve number of matching nodes for given user input.
        :param user_input: string specifies search value which is applied over all property entries of each node
        :param filtered_labels: list contains labels which match the type of node
        :param filtered_properties: list contains property identifiers which are contained in matched nodes
        :param is_contemporary: filter as int (None, 0, 1) if entry contains a contemporary (Person)
        :param include_unknown_period: filter as int (None, 0, 1) if node with period="unknown" is included
        :param filteredStartyear: int for year as start point in time to restrict the search space
        :param filteredEndyear: int for year as end point in time to restrict the search space
        :param takeAsPeriod: filter as int (None, 0, 1) if Startyear and Endyear should bei treated  strict or period
        :param longitude: float for longitude
        :param latitude: float for latitude
        :param include_unknown_location: filter as int (None, 0, 1) if node with location="unknown" is included
        :param tolance_or_radius: float for radius or tolerance to specify the search space in database coordinates
        :return: int specifies the number of matching nodes
        """
        with GraphConnectionManager(self.url, self.username, self.password) as graph:

            query = self._combine_return_line_and_query_GeoFilter(user_input, search_in_all_properties,
                                                                  filtered_labels, filtered_properties, is_contemporary,
                                                                  include_unknown_period, takeAsPeriod,
                                                                  filteredStartyear, filteredEndyear, latitude,
                                                                  longitude, include_unknown_location)

            if "count" in query.lower():
                counter = graph.connection.run(query).evaluate()
                return counter
            else:
                results = graph.connection.run(query)

            near_coordinates = set()
            if longitude or latitude:
                near_coordinates = self._test_for_near_locations_byCoords(latitude, longitude, tolance_or_radius)
                self.logger.info("Longitude or Latitude available.")
                self.logger.info(str(len(near_coordinates)) + " coordinates in range retrieved from db")

            node_count = 0
            for res in results:

                subcats = res.get("n.subclasses")
                time_info = res.get("n.period")
                locations = res.get("n.locations")

                include_node = self._filter_pipeline_Geo(is_contemporary, include_unknown_period, takeAsPeriod,
                                                         filteredStartyear, filteredEndyear,
                                                         near_coordinates, include_unknown_location,
                                                         longitude, latitude, subcats,
                                                         time_info, locations)
                if include_node:
                    node_count += 1
            return int(node_count)

    def retrieve_nodes_by_input(self, user_input: str, search_in_all_properties: bool, filtered_labels: List,
                                is_contemporary: int, filtered_properties: List, takeAsPeriod: int,
                                include_unknown_period: int, filteredStartyear: int, filteredEndyear: int) -> List:
        """
        Retrieve list of reduced representation of matching nodes for given user input.
        :param user_input: string specifies search value which is applied over all property entries of each node
        :param filtered_labels: list contains labels which match the type of node
        :param filtered_properties: list contains property identifiers which are contained in matched nodes
        :return: list contains reduced representation of nodes as dict
        """
        with GraphConnectionManager(self.url, self.username, self.password) as graph:

            return_line = self._returnline_for_node_retrieval()

            query = self._build_query(return_line, user_input, search_in_all_properties,
                                      filtered_labels, filtered_properties)

            results = graph.connection.run(query)
            matching_nodes = []
            for res in results:

                n = res.get("n")
                node = {"label": list(n.labels)}
                properties = dict(n)
                node["uri"] = properties.pop("uri")
                node.update(self._map_properties(node["label"], properties))

                del res

                include_node = self._filter_pipeline_person_and_time(is_contemporary, include_unknown_period,
                                                                     takeAsPeriod, filteredStartyear, filteredEndyear,
                                                                     node["subcategories"], node["period"])
                if include_node:
                    node["locations"] = self._get_location_Continent_Country_ByCoords(n["locations"])
                    node = self._get_image_urls_for_node(node)

                    matching_nodes.append(node)
            return matching_nodes

    def retrieve_nodes_by_uri_and_radius(self, uri:str, tolerance_or_radius:float) -> List:

        start_node = self.retrieve_node_by_uri(uri)

        with GraphConnectionManager(self.url, self.username, self.password) as graph:

            return_line = self._returnline_for_node_retrieval(location_param=f", n.locations")

            query = f"""
                    MATCH (n)
                    {return_line}
                    """
            results = graph.connection.run(query)

            locations = start_node.get("locations")
            if "unknown" in locations:
                return []

            locations = self._reformat_locations(locations)
            near_coordinates = []
            for coord in locations:
                latitude = coord[0]
                longitude = coord[1]
                near_coordinates += list(self._test_for_near_locations_byCoords(latitude, longitude, tolerance_or_radius))

            near_coordinates = set(near_coordinates)
            matching_nodes = []
            coord_dict = {}
            for res in results:

                n = res.get("n")
                node = {"label": list(n.labels)}
                properties = dict(n)
                node["uri"] = properties.pop("uri")
                node.update(self._map_properties(node["label"], properties))

                locations = res.get("n.locations")
                if "unknown" in locations:
                    continue
                coords = self._reformat_locations(locations)
                include_node = self._test_if_coords_are_in_search_space(coords, near_coordinates)

                if include_node:
                    node["locations"] = self._get_location_Continent_Country_ByCoords(locations)
                    node = self._get_image_urls_for_node(node)
                    matching_nodes.append(node)
            return matching_nodes

    #pipelines for decision of node inclusion
    def _filter_pipeline_person_and_time(self, is_contemporary:int, include_unknown_period:int, takeAsPeriod:int,
                                         filteredStartyear:int, filteredEndyear:int, subcats, time_info) -> bool:

        if not is_contemporary and include_unknown_period == 1 and not takeAsPeriod \
                and not filteredStartyear and not filteredEndyear:
            include_node = True
        else:
            include_node = self._filter_is_contemporary(is_contemporary, subcats)
            if not include_node:
                return include_node

            include_node = self._filter_time_information(time_info, filteredStartyear, filteredEndyear,
                                                         takeAsPeriod, include_unknown_period)

        return include_node

    def _filter_pipeline_Geo(self, is_contemporary: int, include_unknown_period: int, takeAsPeriod: int,
                                   filteredStartyear: int, filteredEndyear: int, coordinatesInSearchedRange: set,
                                   include_unknown_location: int, longitude: float, latitude: float, subcats,
                                   time_info, locations) -> bool:

        include_node = self._filter_pipeline_person_and_time(is_contemporary, include_unknown_period, takeAsPeriod,
                                                             filteredStartyear, filteredEndyear, subcats, time_info)

        if include_node and "unknown" in locations:
            include_node = self._test_for_unkown(include_unknown_location)
        elif include_node and (longitude or latitude):
            include_node = self._filter_location(locations, coordinatesInSearchedRange)

        return include_node

    def _filter_pipeline_GeoByName(self, is_contemporary:int, include_unknown_period:int, takeAsPeriod:int,
                                   filteredStartyear:int, filteredEndyear:int,
                                   include_unknown_location:int, search_regions:list, subcats,
                                   time_info, locations) -> bool:

        include_node = self._filter_pipeline_person_and_time(is_contemporary, include_unknown_period, takeAsPeriod,
                                                                 filteredStartyear, filteredEndyear, subcats, time_info)

        if include_node:
            include_node = self._filter_location_ByName(locations, search_regions, include_unknown_location)

        return include_node

    def _concat_continents_countries(self, continents:List, countries:List) -> list:

        if not continents:
            continents = []
        if not countries:
            countries = []

        if self._list_is_empty(continents) and self._list_is_empty(countries):
            return None

        self.logger.info(str(continents))
        self.logger.info(str(countries))
        search_regions = list(set(continents))
        self.logger.info(str(search_regions))
        self.logger.info(str(len(countries)))
        for country in set(countries):
            search_regions.append(country)
        self.logger.info(str(search_regions))

        return search_regions

    def retrieve_counter_nodes_by_inputWithGeoFilterByName(self, user_input: str, search_in_all_properties: bool,
                                                           filtered_labels: List, is_contemporary: int,
                                                           filtered_properties: List, takeAsPeriod: int,
                                                           filteredStartyear: int, filteredEndyear: int,
                                                           include_unknown_period: int, include_unknown_location: int,
                                                           continents: List, countries:List) -> int:
        """
        Retrieve number of matching nodes for given user input.
        :param user_input: string specifies search value which is applied over all property entries of each node
        :param filtered_labels: list contains labels which match the type of node
        :param filtered_properties: list contains property identifiers which are contained in matched nodes
        :param is_contemporary: filter as int (None, 0, 1) if entry contains a contemporary (Person)
        :param include_unknown_period: filter as int (None, 0, 1) if node with period="unknown" is included
        :param filteredStartyear: int for year as start point in time to restrict the search space
        :param filteredEndyear: int for year as end point in time to restrict the search space
        :param takeAsPeriod: filter as int (None, 0, 1) if Startyear and Endyear should bei treated  strict or period
        :param continents: List of Strings
        :param countries: List of Strings
        :param include_unknown_location: filter as int (None, 0, 1) if node with location="unknown" is included
        :return: int specifies the number of matching nodes
        """
        with GraphConnectionManager(self.url, self.username, self.password) as graph:

            query = self._combine_return_line_and_query_GeoFilterByName(user_input, search_in_all_properties,
                                                                        filtered_labels, filtered_properties,
                                                                        is_contemporary, include_unknown_period,
                                                                        takeAsPeriod, filteredStartyear,
                                                                        filteredEndyear, continents, countries,
                                                                        include_unknown_location)

            if "count" in query.lower():
                counter = graph.connection.run(query).evaluate()
                return counter
            else:
                results = graph.connection.run(query)

            search_regions = self._concat_continents_countries(continents, countries)

            node_count = 0
            for res in results:

                include_node = self._filter_pipeline_GeoByName(is_contemporary, include_unknown_period, takeAsPeriod,
                                                               filteredStartyear, filteredEndyear,
                                                               include_unknown_location, search_regions,
                                                               res.get("n.subclasses"),
                                                               res.get("n.period"), res.get("n.locations"))

                if include_node:
                    node_count +=1
            return int(node_count)

    def retrieve_nodes_by_input_with_geoDataByName(self, user_input: str, search_in_all_properties: bool,
                                                   filtered_labels: List, is_contemporary: int,
                                                   filtered_properties: List, takeAsPeriod: int,
                                                   include_unknown_period: int, filteredStartyear: int,
                                                   filteredEndyear: int, include_unknown_location:int,
                                                   continents:List, countries:List) -> List:
        """
        Retrieve list of reduced representation of matching nodes for given user input.
        :param user_input: string specifies search value which is applied over all property entries of each node
        :param filtered_labels: list contains labels which match the type of node
        :param filtered_properties: list contains property identifiers which are contained in matched nodes
        :return: list contains reduced representation of nodes as dict
        """
        with GraphConnectionManager(self.url, self.username, self.password) as graph:

            return_line = self._returnline_for_node_retrieval(location_param=f", n.locations")
            query = self._build_query(return_line, user_input, search_in_all_properties,
                                      filtered_labels, filtered_properties)

            results = graph.connection.run(query)

            search_regions = self._concat_continents_countries(continents, countries)

            matching_nodes = []
            for res in results:

                n = res.get("n")
                node = {"label": list(n.labels)}
                properties = dict(n)
                node["uri"] = properties.pop("uri")
                node.update(self._map_properties(node["label"], properties))
                locations = res.get("n.locations")

                include_node = self._filter_pipeline_GeoByName(is_contemporary, include_unknown_period, takeAsPeriod,
                                                               filteredStartyear, filteredEndyear,
                                                               include_unknown_location, search_regions,
                                                               node["subcategories"], node["period"], locations)

                if include_node:
                    node["locations"] = self._get_location_Continent_Country_ByCoords(locations)
                    node = self._get_image_urls_for_node(node)

                    matching_nodes.append(node)
            return matching_nodes

    def retrieve_nodes_by_input_with_geoData(self, user_input: str, search_in_all_properties: bool,
                                             filtered_labels: List, is_contemporary: int,
                                             filtered_properties: List, takeAsPeriod: int,
                                             include_unknown_period: int, filteredStartyear: int,
                                             filteredEndyear: int, include_unknown_location:int,
                                             latitude:float, longitude:float, tolerance_or_radius:float) -> List:
        """
        Retrieve list of reduced representation of matching nodes for given user input.
        :param user_input: string specifies search value which is applied over all property entries of each node
        :param filtered_labels: list contains labels which match the type of node
        :param filtered_properties: list contains property identifiers which are contained in matched nodes
        :return: list contains reduced representation of nodes as dict
        """
        with GraphConnectionManager(self.url, self.username, self.password) as graph:

            return_line = self._returnline_for_node_retrieval(location_param=f", n.locations")
            query = self._build_query(return_line, user_input, search_in_all_properties,
                                      filtered_labels, filtered_properties)

            results = graph.connection.run(query)

            near_coordinates = set()
            if longitude or latitude:
                near_coordinates = self._test_for_near_locations_byCoords(latitude, longitude, tolerance_or_radius)
                if near_coordinates:
                    # self.logger.info("%s Longitude or Latitude available.")
                    self.logger.info(str(len(near_coordinates)) + " %s coordinates in range retrieved from db")

            matching_nodes = []
            for res in results:

                n = res.get("n")
                node = {"label": list(n.labels)}
                properties = dict(n)
                node["uri"] = properties.pop("uri")
                node.update(self._map_properties(node["label"], properties))
                locations = res.get("n.locations")
                include_node = self._filter_pipeline_Geo(is_contemporary, include_unknown_period, takeAsPeriod,
                                                               filteredStartyear, filteredEndyear,
                                                               near_coordinates, include_unknown_location,
                                                               longitude, latitude, node["subcategories"],
                                                               node["period"], locations)
                if include_node:
                    node["locations"] = self._get_location_Continent_Country_ByCoords(locations)
                    node = self._get_image_urls_for_node(node)

                    matching_nodes.append(node)
            return matching_nodes


    def retrieve_uris_by_input(self, labels: list, threshold: int, user_input="") -> List:
        """
        Retrieve list of matching node uris for given user input.
        :param label: string contains label which matches the type of node
        :param user_input: string specifies search value which is applied over all uris and node names
        :param threshold: limit the number of returning uris
        :return: list contains uris which match the user input
        """
        with GraphConnectionManager(self.url, self.username, self.password) as graph:

            input_filter = ""
            if user_input:
                user_input = user_input.lower()
                # 28 is the length of uri preamble 'http://dbpedia.org/resource/'
                input_filter = f'substring(toLower(toString(n.uri)), 28) CONTAINS "{user_input}" OR ANY(x IN n.name WHERE toLower(toString(x)) CONTAINS "{user_input}")'

            threshold_query_param = self._create_threshold_query_param(threshold)

            label_filter = ""
            if labels:
                label_filter = self._create_label_query_param(labels, True)

            return_line = f"""RETURN DISTINCT n.uri AS uri, n.name AS name
                            ORDER BY uri"""

            if label_filter:
                if input_filter != "":
                    input_filter = f"AND {input_filter}"
                query = f"""
                            MATCH (n)
                            {label_filter}
                            {input_filter}
                            {return_line}
                            {threshold_query_param};
                        """
            else:
                if input_filter != "":
                    input_filter = f"WHERE {input_filter}"
                query = f"""
                        MATCH (n)
                        {input_filter}
                        {return_line}
                        {threshold_query_param};
                        """

            matching_uris = []

            results = graph.connection.run(query)
            for res in results:
                matching_uris.append({"uri": res.get("uri"), "name": res.get("name")})

            return matching_uris

    def retrieve_node_by_uri(self, uri: str) -> OrderedDict:
        """
        Retrieve a specific node from neo4j graph database instance.
        :param uri: string specifies the identifier of node
        :return: dict represents node as description + x format
        """
        with GraphConnectionManager(self.url, self.username, self.password) as graph:
            result = NodeMatcher(graph.connection).match(uri=uri).first()
            node = OrderedDict()
            node["uri"] = uri
            node["label"] = list(result.labels)
            properties = dict(result)
            properties.pop("uri")
            node.update(self._map_properties(node["label"], properties))

            locations = node["locations"]
            node["locations"] = self._get_location_Continent_Country_ByCoords(locations)
            node = self._get_image_urls_for_node(node)

            node["properties"]["abstract"] = ""
            if "dbo:abstract" in node["properties"]:
                node["properties"]["abstract"] = node["properties"]["dbo:abstract"][0]
            elif "dbo:comment" in node["properties"] and not "dbo:abstract" in node["properties"]:
                node["properties"]["abstract"] = node["properties"]["dbo:comment"][0]

            return node

    def retrieve_node_location_names_by_uri(self, uri: str) -> OrderedDict:
        """
        Retrieve a specific node from neo4j graph database instance.
        :param uri: string specifies the identifier of node
        :return: dict represents node as description + x format
        """
        with GraphConnectionManager(self.url, self.username, self.password) as graph:
            result = NodeMatcher(graph.connection).match(uri=uri).first()
            properties = dict(result)
            return {"node_locations": self._get_location_Continent_Country_ByCoords(properties["locations"])}

    def retrieve_location_names_by_coordinate(self, latitude:float, longitude:float) -> OrderedDict:
        return {"node_locations": self._get_location_Continent_Country_ByCoords([(latitude, longitude)])}


    def retrieve_all_nodes(self, threshold: int) -> List:
        """
        Retrieve list of all nodes from neo4j graph database instance.
        :param threshold: limit the number of returned nodes
        :return: list contains nodes which correspond to description + x format
        """
        with GraphConnectionManager(self.url, self.username, self.password) as graph:
            nodes = []

            threshold_query_param = self._create_threshold_query_param(threshold)

            query = f"""
            MATCH (n) RETURN n {threshold_query_param}
            """
            results = graph.connection.run(query)
            for res in results:
                n = res.get("n")
                node = {"label": list(n.labels)}
                properties = dict(n)
                node["uri"] = properties.pop("uri")
                node.update(self._map_properties(node["label"], properties))
                node = self._get_image_urls_for_node()

                nodes.append(node)
            return nodes

    def retrieve_node_connections(self, uri: str, labels: List, threshold: int,
                                  transform_to_similarity = False,
                                  retrieve_semantic_scores = True,
                                  semantic_scores_between_targets= False,
                                  source_target_score_dict = {}) -> List:
        """
        Retrieve list of relations of a specific node.
        :param uri: string specifies the identifier of node
        :param labels: list specifies labels which comply with those of the matching nodes
        :param threshold: limit the number of returned relations
        :return: list contains reduced dict representation of a relation
        """
        with GraphConnectionManager(self.url, self.username, self.password) as graph:

            return_line = self._returnline_for_connections(threshold, location_param=f", m.locations", n=1)
            query = self._build_query_connections(uri, return_line, None, None,
                                                  labels, None)

            results = graph.connection.run(query)

            source_node = self.retrieve_node_by_uri(uri)

            relations = []
            for res in results:

                rel = res.get("r")
                node = res.get("m")

                if node.get("uri") == uri:
                    self.logger.info("uri is the same as source")
                    continue

                target = self.retrieve_node_by_uri(node.get("uri"))

                weights = {"unitDistance": 1, "semantic_distance": -1}  # use for semantic distance
                if retrieve_semantic_scores:
                    target, weights, source_target_score_dict = self._put_semantic_score_into_target(uri,
                                                                                                     source_target_score_dict,
                                                                                                     target, weights,
                                                                                                     source_node,
                                                                                                     semantic_scores_between_targets,
                                                                                                     transform_to_similarity)

                rel_object = self._create_relationship_object(source=uri, target=target, name=type(rel).__name__,
                                                              properties=self._reformat_relation_properties(dict(rel)),
                                                              weights=weights)
                relations.append(rel_object)


            self.logger.info("number of connections: " + str(len(relations)))
            if retrieve_semantic_scores:
                return sorted(relations, key=lambda x: x["name"]), source_target_score_dict
            return sorted(relations, key=lambda x: x["name"])

    def retrieve_nodes_by_labels(self, labels: List, threshold: int) -> List:
        """
        Retrieve list of nodes which labels match any value from a given list.
        :param labels: list specifies labels which comply with those of the matching nodes
        :param threshold: limit the number of returned nodes
        :return: list contains nodes which correspond to description + x format
        """
        with GraphConnectionManager(self.url, self.username, self.password) as graph:

            return_line = self._returnline_for_node_retrieval()
            threshold_query_param = self._create_threshold_query_param(threshold)
            return_line += f""" {threshold_query_param}"""
            query = self._build_query(return_line, None, None, labels, None)

            nodes = []

            results = graph.connection.run(query)
            for res in results:
                n = res.get("n")
                node = {"label": list(n.labels)}
                properties = dict(n)
                node["uri"] = properties.pop("uri")
                node.update(self._map_properties(node["label"], properties))
                node = self._get_image_urls_for_node(node)

                nodes.append(node)
            self.logger.info("number of nodes: "+str(len(nodes)))
            return nodes

    def retrieve_node_connectionsGeoFilterByName(self, uri: str, threshold:int, user_input: str,
                                                 search_in_all_properties: bool, filtered_labels: List,
                                                 is_contemporary: int, filtered_properties: List,
                                                 takeAsPeriod: int, include_unknown_period: int,
                                                 filteredStartyear: int, filteredEndyear: int,
                                                 include_unknown_location:int, continents:List,
                                                 countries:List, semantic_scores_between_targets = False,
                                                 transform_to_similarity = False) -> List:

        """
        Retrieve list of relations of a specific node.
        :param uri: string specifies the identifier of node
        :param labels: list specifies labels which comply with those of the matching nodes
        :param threshold: limit the number of returned relations
        :return: list contains reduced dict representation of a relation
        """
        with GraphConnectionManager(self.url, self.username, self.password) as graph:

            return_line = self._returnline_for_connections(threshold, location_param=f", m.locations")
            query = self._build_query_connections(uri, return_line, user_input, search_in_all_properties,
                                                  filtered_labels, filtered_properties)

            results = graph.connection.run(query)

            search_regions = self._concat_continents_countries(continents, countries)

            source_node = self.retrieve_node_by_uri(uri)
            relations = []
            source_target_score_dict = {}
            for res in results:

                if len(relations) == threshold:
                    self.logger.info("Threshold stop")
                    self.logger.info("number of connections: " + str(len(relations)))
                    if semantic_scores_between_targets and source_target_score_dict!={}:
                        semantic_score_obj = self.get_semantic_rel_object(source_target_score_dict, transform_to_similarity)
                    else:
                        semantic_score_obj = []
                    return sorted(relations, key=lambda x: x["name"]), semantic_score_obj

                rel = res.get("r")
                node = res.get("m")

                if node.get("uri") == uri:
                    self.logger.info("uri is the same as source")
                    continue

                target = self.retrieve_node_by_uri(node.get("uri"))

                weights = {"unitDistance": 1, "semantic_distance":-1}  # use for semantic distance
                include_node = self._filter_pipeline_GeoByName(is_contemporary, include_unknown_period,
                                                               takeAsPeriod, filteredStartyear, filteredEndyear,
                                                               include_unknown_location,
                                                               search_regions, dict(target)["subcategories"],
                                                               dict(target)["period"], res.get("m.locations"))
                if include_node:
                    target, weights, source_target_score_dict = self._put_semantic_score_into_target(source_node["uri"],
                                                                                                     source_target_score_dict,
                                                                                                     target, weights,
                                                                                                     source_node,
                                                                                                     semantic_scores_between_targets,
                                                                                                     transform_to_similarity)
                    rel_object = self._create_relationship_object(source=uri, target=target, name=type(rel).__name__,
                                                                  properties=self._reformat_relation_properties(dict(rel)),
                                                                  weights=weights)
                    relations.append(rel_object)

            if semantic_scores_between_targets and source_target_score_dict!={}:
                semantic_score_obj = self.get_semantic_rel_object(source_target_score_dict, transform_to_similarity)
            else:
                semantic_score_obj = []
            self.logger.info("number of connections: " + str(len(relations)))
            return sorted(relations, key=lambda x: x["name"]), semantic_score_obj

    def retrieve_counter_node_connectionsGeoFilterByName(self, uri: str, threshold: int, user_input: str,
                                                 search_in_all_properties: bool, filtered_labels: List,
                                                 is_contemporary: int, filtered_properties: List,
                                                 takeAsPeriod: int, include_unknown_period: int,
                                                 filteredStartyear: int, filteredEndyear: int,
                                                 include_unknown_location: int, continents: List,
                                                 countries: List) -> int:

        """
        Retrieve list of relations of a specific node.
        :param uri: string specifies the identifier of node
        :param labels: list specifies labels which comply with those of the matching nodes
        :param threshold: limit the number of returned relations
        :return: list contains reduced dict representation of a relation
        """
        with GraphConnectionManager(self.url, self.username, self.password) as graph:

            return_line = self._returnline_for_connections(threshold, location_param=f", m.locations")
            query = self._build_query_connections(uri, return_line, user_input, search_in_all_properties,
                                                  filtered_labels, filtered_properties)
            results = graph.connection.run(query)

            search_regions = self._concat_continents_countries(continents, countries)
            node_count = 0
            
            for res in results:

                if node_count == threshold:
                    return int(node_count)

                node = res.get("m")
                target = self.retrieve_node_by_uri(node.get("uri"))
                include_node = self._filter_pipeline_GeoByName(is_contemporary, include_unknown_period,
                                                               takeAsPeriod, filteredStartyear, filteredEndyear,
                                                               include_unknown_location,
                                                               search_regions, dict(target)["subcategories"],
                                                               dict(target)["period"], res.get("m.locations"))

                if include_node:
                    node_count += 1
            return int(node_count)

    def _get_abstract_from_properties(self, properties) -> str:

        if "dbo:abstract" in properties:
            abstract = properties["dbo:abstract"][0]
        elif "dbo:comment" in properties and not "dbo:abstract" in properties:
            abstract = properties["dbo:comment"][0]
        else:
            abstract = ""

        return abstract


    def _convert_to_json_list(self, semantic_score_list):

        for i in range(len(semantic_score_list)):
            for j in range(len(semantic_score_list[i])):
                if semantic_score_list[i][j] == np.inf:
                    semantic_score_list[i][j] = -1
                    semantic_score_list[j][i] = -1

        return semantic_score_list

    def get_semantic_rel_object(self, semantic_dict, transform_to_similarity):
        semantic_score_obj = []
        for uri in sorted(semantic_dict):
            target_abstracts = semantic_dict[uri]["target_abstracts"]
            del semantic_dict[uri]["target_abstracts"]
            if semantic_dict[uri] not in semantic_score_obj:
                semantic_scores = self.semantic_functions.get_semantic_score_for_all_combination_of_texts(
                                                                                    target_abstracts,
                                                                                    transform_to_similarity)

                semantic_dict[uri]["semantic_scores"] = self._convert_to_json_list(semantic_scores)
                semantic_score_obj.append(semantic_dict[uri])
        return semantic_score_obj

    def _put_semantic_score_into_target(self, uri, score_dict, target, weights, source_node,
                                        semantic_scores_between_targets,
                                        transform_to_similarity):
        if semantic_scores_between_targets and uri not in score_dict:
            score_dict[uri] = {"source": uri, "target_uris": [],
                                             "target_abstracts": [],
                                             "semantic_scores": []}

        target_abstract = dict(target)["properties"]["abstract"]
        source_abstract = dict(source_node)["properties"]["abstract"]

        if score_dict != {}:
            score_dict[uri]["target_uris"].append(dict(target)["uri"])
            score_dict[uri]["target_abstracts"].append(target_abstract)

        semantic_score = self.semantic_functions.get_semantic_score_for_two_texts(source_abstract,
                                                                                  target_abstract,
                                                                                  transform_to_similarity)
        weights["semantic_distance"] = semantic_score if semantic_score != np.inf else -1
        return target, weights, score_dict

    def retrieve_node_connectionsWithoutGeoFilter(self, uri: str, threshold:int, user_input: str,
                                                 search_in_all_properties: bool, filtered_labels: List,
                                                 is_contemporary: int, filtered_properties: List,
                                                 takeAsPeriod: int, include_unknown_period: int,
                                                 filteredStartyear: int, filteredEndyear: int,
                                                  semantic_scores_between_targets = False,
                                                  transform_to_similarity = False) -> List:

        """
            Retrieve list of relations of a specific node.
            :param uri: string specifies the identifier of node
            :param labels: list specifies labels which comply with those of the matching nodes
            :param threshold: limit the number of returned relations
            :return: list contains reduced dict representation of a relation
        """
        with GraphConnectionManager(self.url, self.username, self.password) as graph:

            return_line = self._returnline_for_connections(threshold, location_param=f", m.locations")
            query = self._build_query_connections(uri, return_line, user_input, search_in_all_properties,
                                                  filtered_labels, filtered_properties)

            results = graph.connection.run(query)

            source_node = self.retrieve_node_by_uri(uri)

            relations=[]
            source_target_score_dict = {}
            for res in results:
                if len(relations) == threshold:
                    self.logger.info("Threshold stop")
                    self.logger.info("number of connections: " + str(len(relations)))
                    if semantic_scores_between_targets and source_target_score_dict!={}:
                        semantic_score_obj = self.get_semantic_rel_object(source_target_score_dict, transform_to_similarity)
                    else:
                        semantic_score_obj = []

                    return sorted(relations, key=lambda x: x["name"]), semantic_score_obj

                rel = res.get("r")
                node = res.get("m")

                if node.get("uri") == uri:
                    self.logger.info("uri is the same as source")
                    continue

                target = self.retrieve_node_by_uri(node.get("uri"))
                weights = {"unitDistance": 1, "semantic_distance": -1}  # use for semantic distance
                include_node = self._filter_pipeline_person_and_time(is_contemporary, include_unknown_period,
                                                                     takeAsPeriod, filteredStartyear, filteredEndyear,
                                                                     dict(target)["subcategories"],
                                                                     dict(target)["period"])
                if include_node:

                    target, weights, source_target_score_dict = self._put_semantic_score_into_target(source_node["uri"],
                                                                                            source_target_score_dict,
                                                                                            target, weights,
                                                                                            source_node,
                                                                                            semantic_scores_between_targets,
                                                                                            transform_to_similarity)
                    rel_object = self._create_relationship_object(source=uri, target=target, name=type(rel).__name__,
                                                                  properties=self._reformat_relation_properties(dict(rel)),
                                                                  weights=weights)
                    relations.append(rel_object)

            if semantic_scores_between_targets and source_target_score_dict!={}:
                semantic_score_obj = self.get_semantic_rel_object(source_target_score_dict, transform_to_similarity)
            else:
                semantic_score_obj = []

            self.logger.info("number of connections: " + str(len(relations)))
            return sorted(relations, key=lambda x: x["name"]), semantic_score_obj

    def retrieve_counter_node_connectionsWithoutGeoFilter(self, uri: str, threshold:int, user_input: str,
                                                 search_in_all_properties: bool, filtered_labels: List,
                                                 is_contemporary: int, filtered_properties: List,
                                                 takeAsPeriod: int, include_unknown_period: int,
                                                 filteredStartyear: int, filteredEndyear: int) -> List:

        """
            Retrieve list of relations of a specific node.
            :param uri: string specifies the identifier of node
            :param labels: list specifies labels which comply with those of the matching nodes
            :param threshold: limit the number of returned relations
            :return: list contains reduced dict representation of a relation
        """
        with GraphConnectionManager(self.url, self.username, self.password) as graph:

            return_line = self._returnline_for_connections(threshold, location_param=f", m.locations")
            query = self._build_query_connections(uri, return_line, user_input, search_in_all_properties,
                                                  filtered_labels, filtered_properties)

            results = graph.connection.run(query)

            node_count = 0
            all_count = 0
            for res in results:

                if node_count == threshold:
                    return int(node_count)

                # rel = res.get("r")
                node = res.get("m")
                target = self.retrieve_node_by_uri(node.get("uri"))
                weights = {"unitDistance": 1}  # TODO: remove because deprecated
                include_node = self._filter_pipeline_person_and_time(is_contemporary, include_unknown_period,
                                                                     takeAsPeriod, filteredStartyear, filteredEndyear,
                                                                     dict(target)["subcategories"],
                                                                     dict(target)["period"])
                if include_node:
                    # rel_object = self._create_relationship_object(source=uri, target=target, name=type(rel).__name__,
                    #                                              properties=self._reformat_relation_properties(
                    #                                                  dict(rel)),
                    #                                              weights=weights)
                    node_count += 1
                all_count += 1

            self.logger.info("number of connections: " + str(node_count) + " of "+ str(all_count))
            return int(node_count)

    def retrieve_node_connectionsGeoFilter(self, uri: str, threshold:int, user_input: str,
                                           search_in_all_properties: bool, filtered_labels: List,
                                           is_contemporary: int, filtered_properties: List,
                                           takeAsPeriod: int, include_unknown_period: int,
                                           filteredStartyear: int, filteredEndyear: int,
                                           include_unknown_location:int, longitude:float,
                                           latitude:float, tolerance_or_radius:float,
                                           semantic_scores_between_targets = False,
                                           transform_to_similarity = False) -> List:
        """
        Retrieve list of relations of a specific node.
        :param uri: string specifies the identifier of node
        :param labels: list specifies labels which comply with those of the matching nodes
        :param threshold: limit the number of returned relations
        :return: list contains reduced dict representation of a relation
        """
        with GraphConnectionManager(self.url, self.username, self.password) as graph:

            return_line = self._returnline_for_connections(threshold, location_param=f", m.locations")
            query = self._build_query_connections(uri, return_line, user_input, search_in_all_properties,
                                                  filtered_labels, filtered_properties)

            results = graph.connection.run(query)

            near_coordinates = set()
            if longitude or latitude:
                near_coordinates = self._test_for_near_locations_byCoords(latitude, longitude, tolerance_or_radius)
                # self.logger.info("Longitude or Latitude available.")
                self.logger.info(str(len(near_coordinates)) + " coordinates in range retrieved from db")

            source_node = self.retrieve_node_by_uri(uri)

            relations = []
            source_target_score_dict = {}
            for res in results:

                if len(relations) == threshold:
                    self.logger.info("Threshold stop")
                    self.logger.info("number of connections: " + str(len(relations)))
                    if semantic_scores_between_targets and source_target_score_dict!={}:
                        semantic_score_obj = self.get_semantic_rel_object(source_target_score_dict, transform_to_similarity)
                    else:
                        semantic_score_obj = []
                    return sorted(relations, key=lambda x: x["name"]), semantic_score_obj

                rel = res.get("r")
                node = res.get("m")

                if node.get("uri") == uri:
                    self.logger.info("uri is the same as source")
                    continue

                target = self.retrieve_node_by_uri(node.get("uri"))
                weights = {"unitDistance": 1, "semantic_distance": -1}  # use for semantic distance
                include_node = self._filter_pipeline_Geo(is_contemporary, include_unknown_period, takeAsPeriod,
                                                         filteredStartyear, filteredEndyear,
                                                         near_coordinates, include_unknown_location,
                                                         longitude, latitude, dict(target)["subcategories"],
                                                         dict(target)["period"], res.get("m.locations"))
                if include_node:
                    target, weights, source_target_score_dict = self._put_semantic_score_into_target(source_node["uri"],
                                                                                                     source_target_score_dict,
                                                                                                     target, weights,
                                                                                                     source_node,
                                                                                                     semantic_scores_between_targets,
                                                                                                     transform_to_similarity)

                    rel_object = self._create_relationship_object(source=uri, target=target, name=type(rel).__name__,
                                                                  properties=self._reformat_relation_properties(
                                                                      dict(rel)),
                                                                  weights=weights)
                    relations.append(rel_object)

            if semantic_scores_between_targets and source_target_score_dict!={}:
                semantic_score_obj = self.get_semantic_rel_object(source_target_score_dict, transform_to_similarity)
            else:
                semantic_score_obj = []
            self.logger.info("number of connections: "+ str(len(relations)))
            return sorted(relations, key=lambda x: x["name"]), semantic_score_obj

    def retrieve_counter_node_connectionsGeoFilter(self, uri: str, threshold: int, user_input: str,
                                                 search_in_all_properties: bool, filtered_labels: List,
                                                 is_contemporary: int, filtered_properties: List,
                                                 takeAsPeriod: int, include_unknown_period: int,
                                                 filteredStartyear: int, filteredEndyear: int,
                                                 include_unknown_location: int, longitude: float,
                                                 latitude: float, tolerance_or_radius:float) -> List:

        """
        Retrieve list of relations of a specific node.
        :param uri: string specifies the identifier of node
        :param labels: list specifies labels which comply with those of the matching nodes
        :param threshold: limit the number of returned relations
        :return: list contains reduced dict representation of a relation
        """
        with GraphConnectionManager(self.url, self.username, self.password) as graph:

            return_line = self._returnline_for_connections(threshold, location_param=f", m.locations")
            query = self._build_query_connections(uri, return_line, user_input, search_in_all_properties,
                                                  filtered_labels, filtered_properties)

            results = graph.connection.run(query)

            near_coordinates = set()
            if longitude or latitude:
                near_coordinates = self._test_for_near_locations_byCoords(latitude, longitude, tolerance_or_radius)
                # self.logger.info("Longitude or Latitude available.")
                self.logger.info(str(len(near_coordinates)) + " coordinates in range retrieved from db")
            node_count = 0

            for res in results:

                if node_count == threshold:
                    return int(node_count)

                node = res.get("m")
                target = self.retrieve_node_by_uri(node.get("uri"))
                include_node = self._filter_pipeline_Geo(is_contemporary, include_unknown_period,
                                                               takeAsPeriod, filteredStartyear, filteredEndyear,
                                                               near_coordinates, include_unknown_location,
                                                               longitude, latitude, dict(target)["subcategories"],
                                                               dict(target)["period"], res.get("m.locations"))

                if include_node:
                    node_count += 1
            return int(node_count)

    def retrieve_abstract_for_uri(self, uri) -> str:

        with GraphConnectionManager(self.url, self.username, self.password) as graph:
            result = NodeMatcher(graph.connection).match(uri=uri).first()
            node = OrderedDict()
            node["uri"] = uri
            node["label"] = list(result.labels)
            properties = dict(result)
            properties.pop("uri")
            # add description part
            node.update(self._map_properties(node["label"], properties))
            if "dbo:abstract" in node["properties"]:
                abstract = node["properties"]["dbo:abstract"][0]
            elif "dbo:comment" in node["properties"] and not "dbo:abstract" in node["properties"]:
                abstract = node["properties"]["dbo:comment"][0]
            else:
                abstract = ""
            return abstract

    def retrieve_semantic_scores_for_abstracts(self, abstracts, ref_abstract=None, transform_to_similarity=False):
        if ref_abstract:
            semantic_scores = self.semantic_functions.get_semantic_score_for_reftext_and_listoftexts(ref_abstract,
                                                                                                    abstracts,
                                                                                                    transform_to_similarity)
        elif not ref_abstract and len(abstracts) == 2:
            semantic_scores = api_handler.semantic_functions.get_semantic_score_for_two_texts(abstracts[0], abstracts[1],
                                                                                              transform_to_similarity)
            semantic_scores = [float(semantic_scores)]
        else:
            semantic_scores = api_handler.semantic_functions.get_semantic_score_for_all_combination_of_texts(abstracts,
                                                                                                             transform_to_similarity)

        # json is not able to save numpy.ndarrays so it is converted to list format
        # json is not able to save numpy.floats so it is converted to float
        # if isinstance(semantic_scores, np.ndarray):
        #    semantic_scores = semantic_scores.tolist()
        return semantic_scores

    def retrieve_abstracts_for_list_of_uris(self, list_of_uris:list) -> dict:

        """
                Retrieve dictionary of uri:abstract for a list of uris.
                :param list_of_uris: list of strings specifying the identifiers of nodes
                :return: dictionary of uri:abstract
        """

        with GraphConnectionManager(self.url, self.username, self.password) as graph:
            uri_abstract_list = []
            for uri in list_of_uris:
                result = NodeMatcher(graph.connection).match(uri=uri).first()
                node = OrderedDict()
                node["uri"] = uri
                node["label"] = list(result.labels)
                properties = dict(result)
                properties.pop("uri")
                # add description part
                node.update(self._map_properties(node["label"], properties))
                if "dbo:abstract" in node["properties"]:
                    uri_abstract_list.append(node["properties"]["dbo:abstract"][0])
                elif "dbo:comment" in node["properties"] and not "dbo:abstract" in node["properties"]:
                    uri_abstract_list.append(node["properties"]["dbo:comment"][0])
                else:
                    uri_abstract_list.append("")

        return uri_abstract_list

### API calls here ###
app = Flask(__name__)
cors = CORS(app)
api = Api(app, version="1.0", title="eTaRDiS REST API",
          description="Handle communication between services and Neo4j graph database by applying GET and POST requests and using JSON as provided data format.",
          )
ns_get = api.namespace(name="GET", description="Description of available get operations", path="/")
ns_post = api.namespace(name="POST", description="Description of available post operations", path="/")

neo4j_url = os.getenv("NEO4J_URL")
neo4j_port = os.getenv("NEO4J_BOLT_PORT")
neo4j_user = os.getenv("NEO4J_USER")
neo4j_password = os.getenv("NEO4J_PASSWD")
api_handler = GraphApi(url=f"bolt://{neo4j_url}:{neo4j_port}", username=neo4j_user, password=neo4j_password)
#api_handler = GraphApi(url=f"bolt://0.0.0.0:7687", username="neo4j", password="password")
logger = logging.getLogger(__name__)

#export FLASK_APP=graph_api_develop.py FLASK_ENV=development FLASK_DEBUG=1
#export NEO4J_URL=0.0.0.0 NEO4J_BOLT_PORT=7687 NEO4J_USER=neo4j NEO4J_PASSWD=password
#flask run --host=localhost --port=5000

# definition of endpoints

### request parsers ###
#endpoint / getAvailableCountriesPerContinent
continentParser = reqparse.RequestParser()
continentParser.add_argument("continents", type=str, action="split", required=False, nullable=True,
                                      help="Specify the continents to retrieve its countries \n e.g. one or comma separated "
                                           "combination of available continents",
                                      location="json")

# endpoint /getSingleNode
uriParser = reqparse.RequestParser()
uriParser.add_argument("uri", type=str, required=True,
                       help="Specify the uri to match a node which is considered as starting point \n "
                            "e.g. http://dbpedia.org/resource/Hundred_Years'_War is a possible uri",
                       location="json")


#endpoint /getAllUris
labelParser = reqparse.RequestParser()
labelParser.add_argument("labels", type=str, action="split", required=False, nullable=True,
                          help="Specify the label to which all matched nodes belong to",
                          location="json")
                          #choices=["Agent", "CulturalArtifact", "Event", "MaterialObject", "Miscellaneous", "Place",
                          #         "TimePeriod", "TopicalConcept"],

# endpoint /getMatchingUris
labelInputParser = labelParser.copy()
labelInputParser.add_argument("userInput", type=str, required=True, default="",
                              help="Specify a value to find matching nodes that contain the value in their respective"
                                   " names or URIs \n e.g. sheriff is a possible value",
                              location="json")


#endpoint /getNodesByGeoFilter
uriRadiusParser = uriParser.copy()
uriRadiusParser.add_argument("tolerance_or_radius", type=float, required=False, nullable=True,
                             help="Specify the radius to search (float)",
                             location="json")

# endpoint /getNodesByLabels
labelsThresholdParser = reqparse.RequestParser()
labelsThresholdParser.add_argument("labels", type=str, action="split", required=False, nullable=True,
                                   help="Specify the labels of the related node \n e.g. one or comma separated "
                                        "combination of available labels: Agent, CulturalArtifact, Event, MaterialObject, "
                                        "Miscellaneous, Place, TimePeriod, TopicalConcept",
                                   location="json")
labelsThresholdParser.add_argument("threshold", type=int, required=False,
                                   help="Specify the maximum number of connections which should be retrieved for each"
                                        " node \n e.g. 10 is a possible value",
                                   location="json")
labelsThresholdParser.add_argument("semantic_scores_between_targets", type=int, required=False, nullable=True,
                                   help="Specify if semantic distance is transformed to"
                                        " a similarity score"
                                        " - as boolean -> 1:True, 0: False null:False",
                                   location="json")
labelsThresholdParser.add_argument("transform_to_similarity", type=int, required=False, nullable=True,
                                   help="Specify if semantic distance is transformed to a similarity score"
                                         " - as boolean -> 1:True, 0: False null:False",
                                   location="json")

# endpoint /getConnectionsForNode
uriLabelsThresholdParser = labelsThresholdParser.copy() #reqparse.RequestParser()
uriLabelsThresholdParser.add_argument("uri", type=str, required=True,
                                      help="Specify the uri to match a node which is considered as starting point \n "
                                           "e.g. http://dbpedia.org/resource/Hundred_Years'_War is a possible uri",
                                      location="json")

# endpoint /getAllNodes
thresholdNodesParser = reqparse.RequestParser()
thresholdNodesParser.add_argument("thresholdNodes", type=int, required=False, nullable=True,
                                  help="Specify the maximum number of nodes which should be retrieved \n "
                                       "e.g. 100 is a possible value",
                                  location="json")


# endpoint /getAllNodesAndConnections
thresholdParser = thresholdNodesParser.copy()
thresholdParser.add_argument("thresholdConnections", type=int, required=False, nullable=True,
                             help="Specify the maximum number of connections which should be retrieved for each node "
                                  "\n e.g. 10 is a possible value",
                             location="json")

# endpoint semanticDistanceMatrixModel
urilistParser = reqparse.RequestParser()
urilistParser.add_argument("ref_uri", type=str, required=False, nullable=True,
                           help="Specify a reference uri that should be evaluated in semantic distance according"
                                "to a list of uris.",
                           location="json")
urilistParser.add_argument("uris", type=str, action="split",  required=False, nullable=True,
                           help="Specify the list of uris \n e.g. one or comma separated ",
                           location="json")
urilistParser.add_argument("transform_to_similarity", type=int, required=False, nullable=True,
                           help="Specify if semantic scores are returned as similarity scores - as boolean -> 1:True,"
                                " 0: False null:True", location="json")

# endpoint /getNodesCountByInput, /getNodesByInput
inputLabelsPropertiesParser = reqparse.RequestParser()
inputLabelsPropertiesParser.add_argument("userInput", type=str, required=False,
                                         help="Specify a value to find matching nodes that contain the value in their "
                                              "respective names or URIs \n e.g. sheriff is a possible value",
                                         location="json")
inputLabelsPropertiesParser.add_argument("include_search_in_all_properties", type=int, required=False, nullable=True,
                                         help="Specify if user_input is searched in name and uri or in all props "
                                              "as boolean",
                                         location="json")
inputLabelsPropertiesParser.add_argument("labels", type=str, action="split",  required=False, nullable=True,
                                         help="Specify the labels of the matching nodes \n e.g. one or comma separated "
                                              "combination of available labels separated by whitespace: Agent,"
                                              " CulturalArtifact, Event, "
                                              "MaterialObject, Miscellaneous, Place, TimePeriod, TopicalConcept",
                                         location="json")
inputLabelsPropertiesParser.add_argument("isContemporary", type=int, required=False, nullable=True,
                                         help="Specify if the results should only include persons "
                                              "as boolean",
                                         location="json")
inputLabelsPropertiesParser.add_argument("properties", type=str, action="split", required=False, nullable=True,
                                         help="Specify the properties which should be contained in matching nodes \n "
                                              "e.g. one or comma separated combination of available properties such "
                                              "as dbo:commander",
                                         location="json")
inputLabelsPropertiesParser.add_argument("start_year", type=int, required=False, nullable=True,
                                         help="Specify the start of the timespan as year (int)",
                                         location="json")
inputLabelsPropertiesParser.add_argument("end_year", type=int, required=False, nullable=True,
                                         help="Specify the end of the timespan as year (int)",
                                         location="json")
inputLabelsPropertiesParser.add_argument("takeAsPeriod", type=int, required=False, nullable=True,
                                         help="Specify if start end end year are "
                                              "taken as a period or as exact time points as boolean",
                                         location="json")
inputLabelsPropertiesParser.add_argument("include_unknown_period", type=int, required=False, nullable=True,
                                         help="Specify if period can have the value unknown - as boolean -> 1:True,"
                                              " 0: False null:True",
                                         location="json")


inputLabelsPropertiesParserWithoutGeo = inputLabelsPropertiesParser.copy()
inputLabelsPropertiesParserWithoutGeo.add_argument("threshold", type=int, required=False,
                                                   help="Specify the maximum number of connections which should be "
                                                        "retrieved for each node \n e.g. 10 is a possible value",
                                                   location="json")
inputLabelsPropertiesParserWithoutGeo.add_argument("uri", type=str, required=True,
                                                   help="Specify the uri to match a node which is considered as "
                                                        "starting point \n e.g. http://dbpedia.org/resource/"
                                                        "Hundred_Years'_War is a possible uri",
                                                   location="json")
inputLabelsPropertiesParserWithoutGeo.add_argument("semantic_scores_between_targets", type=int, required=False, nullable=True,
                                                    help="Specify if semantic distance is transformed to"
                                                         " a similarity score"
                                                         " - as boolean -> 1:True, 0: False null:False",
                                                    location="json")
inputLabelsPropertiesParserWithoutGeo.add_argument("transform_to_similarity", type=int, required=False, nullable=True,
                                                    help="Specify if semantic distance is transformed to"
                                                         " a similarity score"
                                                         " - as boolean -> 1:True, 0: False null:False",
                                                    location="json")

GeoParser = reqparse.RequestParser()
GeoParser.add_argument("latitude", type=float, required=False, nullable=True,
                       help="Specify the latitude (float)", location="json")
GeoParser.add_argument("longitude", type=float, required=False, nullable=True, help="Specify the longitude (float)",
                       location="json")

# endpoint /getNodesByInputWihtGeoFilter
inputLabelsPropertiesGeoParser = inputLabelsPropertiesParser.copy()
inputLabelsPropertiesGeoParser.add_argument("include_unknown_location", type=int, required=False, nullable=True,
                                            help="Specify if location can have the value unknown - as boolean "
                                                 "-> 1:True, 0: False null:True", location="json")
inputLabelsPropertiesGeoParser.add_argument("latitude", type=float, required=False, nullable=True,
                                            help="Specify the latitude (float)", location="json")
inputLabelsPropertiesGeoParser.add_argument("longitude", type=float, required=False, nullable=True,
                                            help="Specify the longitude (float)",
                                            location="json")
inputLabelsPropertiesGeoParser.add_argument("tolerance_or_radius", type=float, required=False, nullable=True,
                                            help="Specify the radius/tolerance to search (float)",
                                            location="json")

inputLabelsPropertiesGeoParserConnections = inputLabelsPropertiesGeoParser.copy()
inputLabelsPropertiesGeoParserConnections.add_argument("threshold", type=int, required=False,
                                                       help="Specify the maximum number of connections which "
                                                            "should be retrieved for each node \n e.g. 10 is "
                                                            "a possible value", location="json")
inputLabelsPropertiesGeoParserConnections.add_argument("uri", type=str, required=True,
                                                       help="Specify the uri to match a node which is considered as "
                                                            "starting point \n e.g. http://dbpedia.org/resource/"
                                                            "Hundred_Years'_War is a possible uri",
                                                       location="json")
inputLabelsPropertiesGeoParserConnections.add_argument("semantic_scores_between_targets", type=int, required=False, nullable=True,
                                                       help="Specify if semantic distance is transformed to"
                                                            " a similarity score"
                                                            " - as boolean -> 1:True, 0: False null:False",
                                                       location="json")
inputLabelsPropertiesGeoParserConnections.add_argument("transform_to_similarity", type=int, required=False, nullable=True,
                                                       help="Specify if semantic distance is transformed to"
                                                            " a similarity score"
                                                            " - as boolean -> 1:True, 0: False null:False",
                                                       location="json")

# endpoint /getNodesByInputWihtGeoFilter
inputLabelsPropertiesGeoParserByName = inputLabelsPropertiesParser.copy()
inputLabelsPropertiesGeoParserByName.add_argument("include_unknown_location", type=int, required=False, nullable=True,
                                                  help="Specify if location can have the value unknown - as boolean "
                                                       "-> 1:True, 0: False null:True",
                                                  location="json")
inputLabelsPropertiesGeoParserByName.add_argument("continents", type=str, action="split", required=False, nullable=True,
                                                  help="Specify the latitude (float)",
                                                  location="json")
inputLabelsPropertiesGeoParserByName.add_argument("countries", type=str, action="split", required=False, nullable=True,
                                                  help="Specify the longitude (float)",
                                                  location="json")

inputLabelsPropertiesGeoParserByNameConnections = inputLabelsPropertiesGeoParserByName.copy()
inputLabelsPropertiesGeoParserByNameConnections.add_argument("threshold", type=int, required=False,
                                                             help="Specify the maximum number of connections which "
                                                                  "should be retrieved for each node \n e.g. 10 is "
                                                                  "a possible value",
                                                             location="json")
inputLabelsPropertiesGeoParserByNameConnections.add_argument("uri", type=str, required=True,
                                                             help="Specify the uri to match a node which is considered "
                                                                  "as starting point \n "
                                                                  "e.g. http://dbpedia.org/resource/Hundred_Years'_War "
                                                                  "is a possible uri",
                                                             location="json")
inputLabelsPropertiesGeoParserByNameConnections.add_argument("semantic_scores_between_targets", type=int, required=False, nullable=True,
                                                             help="Specify if semantic distance is transformed to"
                                                                  " a similarity score"
                                                                  " - as boolean -> 1:True, 0: False null:False",
                                                             location="json")
inputLabelsPropertiesGeoParserByNameConnections.add_argument("transform_to_similarity", type=int, required=False, nullable=True,
                                                             help="Specify if semantic distance is transformed to"
                                                                  " a similarity score"
                                                                  " - as boolean -> 1:True, 0: False null:False",
                                                             location="json")

### response models ###
# endpoint /helloWorld
helloWorldModel = api.model("helloWorld Response", {
    "message": fields.String
})

# endpoint /getStatistics
statisticsModel = api.model("getStatistics Response", {
    "available": fields.Boolean(required=True),
    "totalNodes": fields.Integer(),
    "totalRelations": fields.Integer()
})

# endpoint /getTimespan
timespanModel = api.model("getTimespan Response", {
    "timespan": fields.List(fields.Integer)
})

# endpoint /getLocations
locationsModel = api.model("getLocations Response", {
    "locations": fields.List(fields.List(fields.Float))
})

# endpoint /getLabels
labelsModel = api.model("getLabels Response", {
    "labels": fields.List(fields.String)
})

# endpoint /getSubcategories
subcategoryModel = api.model("getSubcategories Response", {
    "subcategories": fields.List(fields.String)
})
#endpoint /getAllUris
uriOnlyEntry = api.model("entry properties uri", {
    "all_uris": fields.List(fields.String)
})

geoNameEntry = api.model("GetAvailableGeoNames Response", {
    "geo_names" : fields.List(fields.String)
})

# endpoint /getProperties
uriEntry = api.model("entry properties uri", {
    "formattedUri": fields.String,
    "uri": fields.String
})
propertiesModel = api.model("getProperties Response", {
    "properties": fields.List(fields.Nested(uriEntry))
})

# endpoint /getMatchingUris
matchingUriEntry = api.model("entry matching uri", {
    "uri": fields.String,
    "name": fields.List(fields.String)
})
matchingUrisModel = api.model("getMatchingUris Response", {
    "matchingUris": fields.List(fields.Nested(matchingUriEntry))
})

LocationEntry = api.model("node location", {
    "continent":fields.String,
    "sub_region":fields.String,
    "country": fields.String,
    "coordinates": fields.List(fields.List(fields.Float))
})

nodeLocations = api.model("node locations", {
    #"uri": fields.String,
    "node_locations":fields.List(fields.Nested(LocationEntry))
})

# endpoint /getSingleNode
wildcard_fields = fields.Wildcard(fields.List(fields.String))
propertyEntry = api.model("entry property values", {
    "*": wildcard_fields
})
nodeEntry = api.model("entry node", {
    "label": fields.List(fields.String),
    "locations": fields.List(fields.Nested(LocationEntry)),
    "name": fields.List(fields.String),
    "period": fields.List(fields.Integer),
    "subcategories": fields.List(fields.String),
    "uri": fields.String,
    "imageUrls":fields.List(fields.String),
    "properties": fields.Nested(propertyEntry)
})
#,    "*": wildcard_fields
#"locations": fields.List(fields.List(fields.Float)),

singleNodeModel = api.model("getSingleNode Response", {
    "node": fields.Nested(nodeEntry)
})

singleNodeLocationsModel = api.model("getSingleNodeLocations Response", {
    "location_names": fields.Nested(nodeLocations)
})


semanticDistanceMatrixModel = api.model("getMatrixWithSMSScores Response", {
    "semantic_distance_scores": fields.List(fields.Float),
    })
# fields.Wildcard,#
# fields.List(fields.List(fields.float)),

# endpoint /getConnectionsForNode
descriptionNodeEntry = api.model("entry description of node", {
    "uri": fields.String,
    "name": fields.List(fields.String),
    "label": fields.List(fields.String),
    "subcategories": fields.List(fields.String),
    "imageUrls": fields.List(fields.String),
    "period": fields.List(fields.Integer),
    "locations": fields.List(fields.List(fields.Float))
})

relationEntry = api.model("entry relation", {
    "source": fields.String,
    "target": fields.Nested(descriptionNodeEntry),
    "name": fields.String,
    "properties": wildcard_fields,
    "weights": wildcard_fields
})

connectionEntry = api.model("entry relation", {
    "source": fields.String,
    "name": fields.String,
    "properties": wildcard_fields,
    "weights": wildcard_fields,
    "target": fields.Nested(nodeEntry)
})

semanticEntry = api.model("entry semantic", {
    "source": fields.String,
    "target_uris": fields.List(fields.String),
    "semantic_scores": fields.List(fields.List(fields.Float))
})

connectionsModel = api.model("getConnectionsForNode Response", {
    "relations": fields.List(fields.Nested(connectionEntry)),
    "semantic_distance_scores": fields.Nested(semanticEntry)
})

# endpoint /getAllNodes
allNodesModel = api.model("getAllNodes Response", {
    "nodes": fields.List(fields.Nested(nodeEntry))
})

# endpoint /getAllNodesAndConnections
allNodesAndConnectionsModel = api.model("getAllNodesAndConnections Response", {
    "nodes": fields.List(fields.Nested(nodeEntry)),
    "relations": fields.List(fields.Nested(relationEntry)),
    "semantic_distance_scores": fields.List(fields.Nested(semanticEntry))
})

# endpoint /getNodesByLabels
nodesByLabelModel = api.model("getNodesByLabels Response", {
    "nodes": fields.List(fields.Nested(nodeEntry))
})

# endpoint /getNodesCountByInput
nodesCountInputModel = api.model("getNodesCountByInput Response", {
    "counter": fields.Integer
})

# endpoint /getNodesByInput
nodesInputModel = api.model("getNodesByInput Response", {
    "matchingNodes": fields.List(fields.Nested(descriptionNodeEntry))
})


@ns_get.route("/helloWorld")
class HelloWorld(Resource):
    @api.response(200, "Success", helloWorldModel)
    @ns_get.doc('helloWorld')
    def get(self):
        """
        Returns a hello world message to test availability of REST API
        """
        return {"message": "Hello World"}


@ns_get.route("/getStatistics")
class Statistics(Resource):
    @api.response(200, "Success", statisticsModel)
    @ns_get.doc('getStatistics')
    def get(self) -> OrderedDict:
        """
        Returns a response which contains status of Neo4j database
                and provides numbers of total available nodes and relations as int
        """
        response = OrderedDict()
        response["available"] = False
        response["totalNodes"] = 0
        response["totalRelations"] = 0
        try:
            total_nodes, total_relations = api_handler.retrieve_statistics()
            response["available"] = True
            response["totalNodes"] = total_nodes
            response["totalRelations"] = total_relations
            status = 200
        except Exception as e:
            logger.error("Could not retrieve statistics: %s", e)
        return response


@ns_get.route("/getLabels")
class Labels(Resource):
    @api.response(200, "Success", labelsModel)
    @ns_get.doc("getlabels")
    def get(self) -> OrderedDict:
        """
        Retrieve a list of available labels as Strings
        """
        response = OrderedDict()
        try:
            response["labels"] = api_handler.retrieve_available_labels()
        except Exception as e:
            logger.error("Could not retrieve labels: %s", e)
        return response


# New Function to get all subcategories, the timespan and coordinates
@ns_get.route("/getSubcategories")
class Subcategories(Resource):
    @api.response(200, "Success", subcategoryModel)
    @ns_get.doc("getSubcategories")
    def get(self) -> OrderedDict:
        """
        Retrieve a list of available Subcategories as Strings
        """
        response = OrderedDict()
        try:
            response["subcategories"] = api_handler.retrieve_available_subcategories()
        except Exception as e:
            logger.error("Could not retrieve labels: %s", e)
        return response


@ns_get.route("/getTimespan")
class Timespan(Resource):
    @api.response(200, "Success", timespanModel)
    @ns_get.doc("getTimespan")
    def get(self) -> OrderedDict:
        """
        Retrieve a list with the year of the most early and the latest node
        """
        response = OrderedDict()
        try:
            response["timespan"] = api_handler.retrieve_timespan()
        except Exception as e:
            logger.error("Could not retrieve labels: %s", e)
        return response


@ns_get.route("/getCoordinates")
class Coordinates(Resource):
    @api.response(200, "Success", locationsModel)
    @ns_get.doc("getCoordinates")
    def get(self) -> OrderedDict:
        """
        Retrieve a list of available coordinates

        if you want to check, which location coordinates specify on the map, you can use
        https://nominatim.openstreetmap.org/ui/reverse.html?lat=22.95736474663196&lon=-101.93844595604823
        """
        response = OrderedDict()
        try:
            response["coordinates"] = api_handler.retrieve_all_coordinates()
        except Exception as e:
            logger.error("Could not retrieve labels: %s", e)
        return response


@ns_get.route("/getProperties")
class Properties(Resource):
    @api.response(200, "Success", propertiesModel)
    @ns_get.doc("getProperties")
    def get(self) -> OrderedDict:
        """
        Retrieve a list of available node properties
        """
        response = OrderedDict()
        try:
            response["properties"] = api_handler.retrieve_available_properties()
        except Exception as e:
            logger.error("Could not retrieve properties: %s", e)
        return response


@ns_get.route("/getAvailableContinents")
class GeoNamesContinents(Resource):
   @api.response(200, "Success", geoNameEntry)
   @ns_get.doc("getAvailableContinents")
   def get(self) -> OrderedDict:
       """
       Retrieve a list of all available continent names
       """
       response = OrderedDict()
       try:
           response["geo_names"] = api_handler.retrieve_available_continents()
       except Exception as e:
           logger.error("Could not retrieve properties: %s", e)
       return response

@ns_get.route("/getAvailableSubRegions")
class GeoNamesSubRegions(Resource):
   @api.response(200, "Success", geoNameEntry)
   @ns_get.doc("getAvailableSubRegions")
   def get(self) -> OrderedDict:
       """
       Retrieve a list of all available continent sub region names
       """
       response = OrderedDict()
       try:
           response["geo_names"] = api_handler.retrieve_available_sub_regions()
       except Exception as e:
           logger.error("Could not retrieve properties: %s", e)
       return response

################POST###########################

def set_default_takeAsPeriod(filtered_takeAsPeriod, filtered_startyear, filtered_endyear):
    if filtered_startyear or filtered_endyear:
        return 1
    else:
        return filtered_takeAsPeriod

@ns_post.route("/getAllUris")
class AllUris(Resource):
    @api.response(200, "Success", uriOnlyEntry)
    @ns_post.doc("getAllUris", parser=labelParser)
    def post(self) -> OrderedDict:
        """
        Retrieve a list of all available uris
        :param selectedLabel: string of Labels, separeted by ','
        :return: List of uris

        Example:
            {
            "labels":"Agent"
            }

        """
        response = OrderedDict()
        try:
            try:
                args = labelParser.parse_args()
                labels = args.get("labels")
            except:
                labels = []
            if not labels:
                labels = []
            response["all_uris"] = api_handler.retrieve_available_uris(labels)
            logger.info("Number of retrieved uris: "+str(len(response["all_uris"])))
        except Exception as e:
            logger.error("Could not retrieve properties: %s", e)
        return response


@ns_post.route("/getAvailableCountriesPerContinent")
class GeoNamesCountries(Resource):
    @api.response(200, "Success", geoNameEntry)
    @ns_post.doc("getAvailableCountriesPerContinent", parser=continentParser)
    def post(self) -> OrderedDict:
        """
        Retrieve a list of all available countries.
        : param continents: string of Continents, separeted by ','
        : return: List of countries

        If parameter "continents" -> List of strings
                is specified, countries are returned for the continents in the list if they are in the geo database

         Look for available continents with GET function "getAvailableContinents"

         Example:
             {
             "continents":"europe,asia"
             }

         Get all available countries with:
            {
            "continents":null
            }

        """
        response = OrderedDict()
        try:
            try:
                args = continentParser.parse_args()
                continents = args.get("continents")
                if not continents:
                    continents = []
            except:
                continents = []

            logger.info(continents)
            response["geo_names"] = api_handler.retrieve_available_countries(continents)
        except Exception as e:
            logger.error("Could not retrieve properties: %s", e)
        return response


@ns_post.route("/getMatchingUris")
class MatchingUris(Resource):
    @api.response(200, "Success", matchingUrisModel)
    @ns_post.doc("getMatchingUris", parser=labelInputParser)
    def post(self) -> OrderedDict:
        """
        Retrieve a list of node uris, which match a list of given labels (optional) and user input
        :param userInput: string
        :param selectedLabel: string of Labels, separeted by ','

        Example parameters:
        {
          "labels": "Agent",
          "userInput": "Mary"
        }

        """
        response = OrderedDict()
        #number_items = 30
        stats = api_handler.retrieve_statistics()
        number_items = stats[0]
        try:
            args = labelInputParser.parse_args()
            labels = args.get("labels")
            user_input = args.get("userInput")

            if user_input:
                user_input = user_input.lower()
            else:
                user_input = ""

            response["matchingUris"] = api_handler.retrieve_uris_by_input(labels, number_items, user_input)
        except Exception as e:
            logger.error("Could not retrieve matching uris: %s", e)
        return response


@ns_post.route("/getContinentCountryForSingleNode")
class ContinentCountryForSingleNode(Resource):
    @api.response(200, "Success", singleNodeLocationsModel)
    @ns_post.doc("get_Continent_Country_single_node", parser=uriParser)
    def post(self) -> OrderedDict:
        """
        Retrieve the geo location names for a given node uri
        : param uri: string
        : return: Node as OrderedDict

        Example parameter (try other uris from "getMatchingUris" or "getAllUris"):
        {
          "uri": "http://dbpedia.org/resource/Mary_of_Waltham"
        }

        """
        response = OrderedDict()
        try:
            args = uriParser.parse_args()
            uri = args.get("uri")
            response["location_names"] = api_handler.retrieve_node_location_names_by_uri(uri)
        except Exception as e:
            logger.error("Could not retrieve single node: %s", e)
        return response

@ns_post.route("/get_Continent_Country_for_Coordinates")
class ContinentCountryForCoordinates(Resource):
    @api.response(200, "Success", singleNodeLocationsModel)
    @ns_post.doc("get_Continent_Country_for_Coordinates", parser=GeoParser)
    def post(self) -> OrderedDict:
        """
        Retrieve the geo location names for a given coordinate as latitude and longitude
        : param latitude: float
        : param longitude: float
        : return: Node as OrderedDict

        Example parameter (try other coordinates from "geAllCoordinates"):
        {
        "latitude": 54.6,
        "longitude": -2
        }

        """
        response = OrderedDict()
        try:
            args = GeoParser.parse_args()
            longitude = args.get("longitude")
            latitude = args.get("latitude")
            response["location_names"] = api_handler.retrieve_location_names_by_coordinate(latitude, longitude)
        except Exception as e:
            logger.error("Could not retrieve single node: %s", e)
        return response

@ns_post.route("/getSingleNode")
class SingleNode(Resource):
    @api.response(200, "Success", singleNodeModel)
    @ns_post.doc("get_single_node", parser=uriParser)
    def post(self) -> OrderedDict:
        """
        Retrieve a node and corresponding properties for a given node uri
        : param uri: string
        : return: Node as OrderedDict

        Example parameter (try other uris from "getMatchingUris" or "getAllUris"):
        {
          "uri": "http://dbpedia.org/resource/Mary_of_Waltham"
        }

        """
        response = OrderedDict()
        try:
            args = uriParser.parse_args()
            uri = args.get("uri")
            response["node"] = api_handler.retrieve_node_by_uri(uri)
        except Exception as e:
            logger.error("Could not retrieve single node: %s", e)
        return response

@ns_post.route("/getConnectionsForNode")
class ConnectionsForNode(Resource):
    @api.response(200, "Success", connectionsModel)
    @ns_post.doc("getConnectionsForNode", parser=inputLabelsPropertiesParserWithoutGeo)
    def post(self) -> OrderedDict:
        """
        Retrieve list of connected nodes for a given node uri
        : params: see example below
        : return: List of relations

        Example parameters (try other uris from "getMatchingUris" or "getAllUris"):
        {
            "userInput": "new",
            "labels":"Agent",
            "properties":"dbo:abstract",
            "include_search_in_all_properties": 1,
            "isContemporary": 1,
            "start_year": 1800,
            "end_year": 2000,
            "takeAsPeriod": 1,
            "include_unknown_period": 0,
            "threshold": 10,
            "uri": "http://dbpedia.org/resource/United_States",
            "semantic_scores_between_targets": 0,
            "transform_to_similarity":0
        }

        """
        response = OrderedDict()
        try:
            args = inputLabelsPropertiesParserWithoutGeo.parse_args()
            user_input = args.get("userInput")
            if user_input:
                user_input = user_input.lower()

            search_in_all_properties = args.get("include_search_in_all_properties")
            if search_in_all_properties:
                search_in_all_properties = api_handler.evaluate_as_bool(search_in_all_properties)

            filtered_labels = args.get("labels")
            is_contemporary = args.get("isContemporary")

            filtered_properties = args.get("properties")

            filtered_takeAsPeriod = args.get("takeAsPeriod")

            filtered_startyear = args.get("start_year")
            filtered_endyear = args.get("end_year")
            if not filtered_takeAsPeriod and not filtered_takeAsPeriod==0:
                filtered_takeAsPeriod = set_default_takeAsPeriod(filtered_takeAsPeriod, filtered_startyear,
                                                                 filtered_endyear)

            include_unknown_period = args.get("include_unknown_period")
            if not include_unknown_period and not include_unknown_period == 0:
                include_unknown_period = api_handler.default_include_unknown

            uri = args.get("uri")
            threshold = args.get("threshold")
            transform_to_similarity = args.get("transform_to_similarity")
            if transform_to_similarity:
                transform_to_similarity = api_handler.evaluate_as_bool(transform_to_similarity)
            else:
                transform_to_similarity = False

            semantic_scores_between_targets = args.get("semantic_scores_between_targets")
            if semantic_scores_between_targets:
                semantic_scores_between_targets = api_handler.evaluate_as_bool(semantic_scores_between_targets)
            else:
                semantic_scores_between_targets = False

            response["semantic_distance_scores"] = []
            relations, semantic_scores = api_handler.retrieve_node_connectionsWithoutGeoFilter(uri,
                                                                                               threshold,
                                                                                               user_input,
                                                                                               search_in_all_properties,
                                                                                               filtered_labels,
                                                                                               is_contemporary,
                                                                                               filtered_properties,
                                                                                               filtered_takeAsPeriod,
                                                                                               include_unknown_period,
                                                                                               filtered_startyear,
                                                                                               filtered_endyear,
                                                                                               semantic_scores_between_targets,
                                                                                               transform_to_similarity)
            response["relations"] = relations
            if len(semantic_scores) == 1:
                response["semantic_distance_scores"] = semantic_scores[0]
            else:
                response["semantic_distance_scores"] = {}

        except Exception as e:
            logger.error("Could not retrieve connections of node: %s", e)
        return response


@ns_post.route("/getConnectionsCounterForNode")
class ConnectionsCounterForNode(Resource):
    @api.response(200, "Success", nodesCountInputModel)
    @ns_post.doc("getConnectionsCounterForNode", parser=inputLabelsPropertiesParserWithoutGeo)
    def post(self) -> OrderedDict:
        """
        Retrieve list of connected nodes for a given node uri
        : params: see example below
        : return: List of relations

        Example parameters (try other uris from "getMatchingUris" or "getAllUris"):
        {
            "userInput": "new",
            "labels":"Agent",
            "properties":"dbo:abstract",
            "include_search_in_all_properties": 1,
            "isContemporary": 1,
            "start_year": 1800,
            "end_year": 2000,
            "takeAsPeriod": 1,
            "include_unknown_period": 0,
              "threshold": 10,
              "uri": "http://dbpedia.org/resource/United_States"
        }

        """
        response = OrderedDict()
        try:
            args = inputLabelsPropertiesParserWithoutGeo.parse_args()
            user_input = args.get("userInput")
            if user_input:
                user_input = user_input.lower()

            search_in_all_properties = args.get("include_search_in_all_properties")
            if search_in_all_properties:
                search_in_all_properties = api_handler.evaluate_as_bool(search_in_all_properties)

            filtered_labels = args.get("labels")
            is_contemporary = args.get("isContemporary")

            filtered_properties = args.get("properties")

            filtered_takeAsPeriod = args.get("takeAsPeriod")

            filtered_startyear = args.get("start_year")
            filtered_endyear = args.get("end_year")
            if not filtered_takeAsPeriod and not filtered_takeAsPeriod==0:
                filtered_takeAsPeriod = set_default_takeAsPeriod(filtered_takeAsPeriod, filtered_startyear,
                                                                 filtered_endyear)

            include_unknown_period = args.get("include_unknown_period")
            if not include_unknown_period and not include_unknown_period == 0:
                include_unknown_period = api_handler.default_include_unknown

            uri = args.get("uri")
            threshold = args.get("threshold")

            response["counter"] = api_handler.retrieve_counter_node_connectionsWithoutGeoFilter(uri, threshold,
                                                                                         user_input,
                                                                                         search_in_all_properties,
                                                                                         filtered_labels,
                                                                                         is_contemporary,
                                                                                         filtered_properties,
                                                                                         filtered_takeAsPeriod,
                                                                                         include_unknown_period,
                                                                                         filtered_startyear,
                                                                                         filtered_endyear)

        except Exception as e:
            logger.error("Could not retrieve connections of node: %s", e)
        return response


@ns_post.route("/getSemanticDistanceScoresForURIList")
class SemanticDistanceScoreMatrix(Resource):
    @api.response(200, "Success", semanticDistanceMatrixModel)
    @ns_post.doc("getSemanticDistanceScoresForURIList", parser=urilistParser)
    def post(self) -> OrderedDict:
        """
        Retrieve semantic distance matrix for a given list of uris
        : params: see example below
        : uris : comma seperated list of uris as strings
        : transform_to_similarity : int, that is interpreted as boolean (null or 0: False, 1: True)
        : return: List of relations

        Note: json only supports list format, so the response matrix with the semantic scores for uris with abstract
              are transformed into a list and have to be reformatted into numpy.ndarrays for further usage

              The scores are in the same order on both axes as the uris with abstracts

        Examples param input:

        {
        "uris":"http://dbpedia.org/resource/Aether_drag_hypothesis,http://dbpedia.org/resource/Austin_International_Drag_Festival,http://dbpedia.org/resource/Canada's_Drag_Race",
        "transform_to_similarity":1
        }
        """
        # start_time = time.time()
        response = OrderedDict()
        try:
            args = urilistParser.parse_args()
            ref_uri = args.get("ref_uri")
            list_of_uris = args.get("uris")
            transform_to_similarity = args.get("transform_to_similarity")
            #logger.info("uris: "+str(list_of_uris))

            response["semantic_distance_scores"] = []
            if not list_of_uris:
                return response

            if transform_to_similarity:
                # "Specify if semantic scores are returned as similarity scores - as boolean -> 1:True,"
                # " 0: False null:False",
                transform_to_similarity = api_handler.evaluate_as_bool(transform_to_similarity)
            else:
                transform_to_similarity = False


            abstracts = api_handler.retrieve_abstracts_for_list_of_uris(list_of_uris)
            #logger.info(abstracts)
            if abstracts == []:
                return response

            if ref_uri:
                ref_abstract = api_handler.retrieve_abstract_for_uri(ref_uri)
            else:
                ref_abstract = None

            response["semantic_distance_scores"] = api_handler.retrieve_semantic_scores_for_abstracts(abstracts,
                                                                                                      ref_abstract,
                                                                                                      transform_to_similarity)


            # json is not able to save numpy.ndarrays so it is converted to list format
            # json is not able to save numpy.floats so it is converted to float
            # if isinstance(semantic_scores, np.ndarray):
            #    semantic_scores = semantic_scores.tolist()

            # end_time = time.time()
            # logger.info("execution time for function: "+str(end_time-start_time))
        except Exception as e:
            logger.error("Could not retrieve connections of node: %s", e)
        return response


@ns_post.route("/getConnectionsForNodeGeoFilterByName")
class ConnectionsForNodeGeoFilterByName(Resource):
    @api.response(200, "Success", connectionsModel)
    @ns_post.doc("getConnectionsForNodeGeoFilterByName", parser=inputLabelsPropertiesGeoParserByNameConnections)
    def post(self) -> OrderedDict:
        """
        Retrieve list of connected nodes for a given node uri
        : params: see example below
        : return: List of relations

        Example parameters (try other uris from "getMatchingUris" or "getAllUris"):
        {
            "userInput": "new",
            "labels":"Agent",
            "properties":"dbo:abstract",
            "include_search_in_all_properties": 1,
            "include_unknown_location": 0,
            "isContemporary": 1,
            "start_year": 1800,
            "end_year": 2000,
            "takeAsPeriod": 1,
            "include_unknown_period": 0,
            "continents": "america",
            "countries": "germany,united kingdom",
              "threshold": 10,
              "uri": "http://dbpedia.org/resource/United_States",
              "semantic_scores_between_targets": 0,
              "transform_to_similarity":0
        }

        """
        response = OrderedDict()
        try:
            args = inputLabelsPropertiesGeoParserByNameConnections.parse_args()
            user_input = args.get("userInput")
            if user_input:
                user_input = user_input.lower()

            search_in_all_properties = args.get("include_search_in_all_properties")
            if search_in_all_properties:
                search_in_all_properties = api_handler.evaluate_as_bool(search_in_all_properties)

            filtered_labels = args.get("labels")
            is_contemporary = args.get("isContemporary")

            filtered_properties = args.get("properties")

            filtered_takeAsPeriod = args.get("takeAsPeriod")

            filtered_startyear = args.get("start_year")
            filtered_endyear = args.get("end_year")
            if not filtered_takeAsPeriod and not filtered_takeAsPeriod==0:
                filtered_takeAsPeriod = set_default_takeAsPeriod(filtered_takeAsPeriod, filtered_startyear,
                                                                 filtered_endyear)

            include_unknown_period = args.get("include_unknown_period")
            if not include_unknown_period and not include_unknown_period == 0:
                include_unknown_period = api_handler.default_include_unknown

            include_unknown_location = args.get("include_unknown_location")
            if not include_unknown_location and not include_unknown_location == 0:
                include_unknown_location = api_handler.default_include_unknown

            continents = args.get("continents")
            countries = args.get("countries")

            uri = args.get("uri")
            threshold = args.get("threshold")
            if not threshold:
                threshold = 0
            transform_to_similarity = args.get("transform_to_similarity")
            if transform_to_similarity:
                transform_to_similarity = api_handler.evaluate_as_bool(transform_to_similarity)
            else:
                transform_to_similarity = False

            semantic_scores_between_targets = args.get("semantic_scores_between_targets")
            if semantic_scores_between_targets:
                semantic_scores_between_targets = api_handler.evaluate_as_bool(semantic_scores_between_targets)
            else:
                semantic_scores_between_targets = False

            response["semantic_distance_scores"] = []
            relations, semantic_scores = api_handler.retrieve_node_connectionsGeoFilterByName(uri, threshold,
                                                                                         user_input,
                                                                                         search_in_all_properties,
                                                                                         filtered_labels,
                                                                                         is_contemporary, filtered_properties,
                                                                                         filtered_takeAsPeriod, include_unknown_period,
                                                                                         filtered_startyear, filtered_endyear,
                                                                                         include_unknown_location, continents,
                                                                                         countries,
                                                                                         semantic_scores_between_targets,
                                                                                         transform_to_similarity)
            response["relations"] = relations
            if len(semantic_scores) == 1:
                response["semantic_distance_scores"] = semantic_scores[0]
            else:
                response["semantic_distance_scores"] = {}

        except Exception as e:
            logger.error("Could not retrieve connections of node: %s", e)
        return response

@ns_post.route("/getConnectionsCounterForNodeGeoFilterByName")
class ConnectionsCounterForNodeGeoFilterByName(Resource):
    @api.response(200, "Success", nodesCountInputModel)
    @ns_post.doc("getConnectionsCounterForNodeGeoFilterByName", parser=inputLabelsPropertiesGeoParserByNameConnections)
    def post(self) -> OrderedDict:
        """
        Retrieve list of connected nodes for a given node uri
        : params: see example below
        : return: List of relations

        Example parameters (try other uris from "getMatchingUris" or "getAllUris"):
        {
            "userInput": "new",
            "labels":"Agent",
            "properties":"dbo:abstract",
            "include_search_in_all_properties": 1,
            "include_unknown_location": 0,
            "isContemporary": 1,
            "start_year": 1800,
            "end_year": 2000,
            "takeAsPeriod": 1,
            "include_unknown_period": 0,
            "continents": "america",
            "countries": "germany,united kingdom",
              "threshold": 10,
              "uri": "http://dbpedia.org/resource/United_States"
        }

        """
        response = OrderedDict()
        try:
            args = inputLabelsPropertiesGeoParserByNameConnections.parse_args()
            user_input = args.get("userInput")
            if user_input:
                user_input = user_input.lower()

            search_in_all_properties = args.get("include_search_in_all_properties")
            if search_in_all_properties:
                search_in_all_properties = api_handler.evaluate_as_bool(search_in_all_properties)

            filtered_labels = args.get("labels")
            is_contemporary = args.get("isContemporary")

            filtered_properties = args.get("properties")

            filtered_takeAsPeriod = args.get("takeAsPeriod")

            filtered_startyear = args.get("start_year")
            filtered_endyear = args.get("end_year")
            if not filtered_takeAsPeriod and not filtered_takeAsPeriod==0:
                filtered_takeAsPeriod = set_default_takeAsPeriod(filtered_takeAsPeriod, filtered_startyear,
                                                                 filtered_endyear)

            include_unknown_period = args.get("include_unknown_period")
            if not include_unknown_period and not include_unknown_period == 0:
                include_unknown_period = api_handler.default_include_unknown

            include_unknown_location = args.get("include_unknown_location")
            if not include_unknown_location and not include_unknown_location == 0:
                include_unknown_location = api_handler.default_include_unknown

            continents = args.get("continents")
            countries = args.get("countries")

            uri = args.get("uri")
            threshold = args.get("threshold")

            response["counter"] = api_handler.retrieve_counter_node_connectionsGeoFilterByName(uri, threshold,
                                                                                         user_input,
                                                                                         search_in_all_properties,
                                                                                         filtered_labels,
                                                                                         is_contemporary, filtered_properties,
                                                                                         filtered_takeAsPeriod, include_unknown_period,
                                                                                         filtered_startyear, filtered_endyear,
                                                                                         include_unknown_location, continents,
                                                                                         countries)

        except Exception as e:
            logger.error("Could not retrieve connections of node: %s", e)
        return response


@ns_post.route("/getConnectionsForNodeGeoFilter")
class ConnectionsForNodeGeoFilter(Resource):
    @api.response(200, "Success", connectionsModel)
    @ns_post.doc("getConnectionsForNodeGeoFilter", parser=inputLabelsPropertiesGeoParserConnections)
    def post(self) -> OrderedDict:
        """
        Retrieve list of connected nodes for a given node uri
        : params: see example below
        : return: List of relations

        Example parameters (try other uris from "getMatchingUris" or "getAllUris"):
        {
            "userInput": "new",
            "labels":"Agent",
            "properties":"dbo:abstract",
            "include_search_in_all_properties": 1,
            "include_unknown_location": 0,
            "isContemporary": 1,
            "start_year": 1800,
            "end_year": 2000,
            "takeAsPeriod": 1,
            "include_unknown_period": 0,
            "latitude": 48.2,
            "longitude": 16.35,
            "tolerance_or_radius":10.0,
              "threshold": 10,
              "uri": "http://dbpedia.org/resource/United_States",
              "semantic_scores_between_targets": 0,
              "transform_to_similarity":0
        }

        """
        response = OrderedDict()
        try:
            args = inputLabelsPropertiesGeoParserConnections.parse_args()
            user_input = args.get("userInput")
            if user_input:
                user_input = user_input.lower()

            search_in_all_properties = args.get("include_search_in_all_properties")
            if search_in_all_properties:
                search_in_all_properties = api_handler.evaluate_as_bool(search_in_all_properties)

            filtered_labels = args.get("labels")
            is_contemporary = args.get("isContemporary")

            filtered_properties = args.get("properties")

            filtered_takeAsPeriod = args.get("takeAsPeriod")

            filtered_startyear = args.get("start_year")
            filtered_endyear = args.get("end_year")
            if not filtered_takeAsPeriod and not filtered_takeAsPeriod==0:
                filtered_takeAsPeriod = set_default_takeAsPeriod(filtered_takeAsPeriod, filtered_startyear,
                                                                 filtered_endyear)

            include_unknown_period = args.get("include_unknown_period")
            if not include_unknown_period and not include_unknown_period == 0:
                include_unknown_period = api_handler.default_include_unknown

            include_unknown_location = args.get("include_unknown_location")
            if not include_unknown_location and not include_unknown_location == 0:
                include_unknown_location = api_handler.default_include_unknown

            longitude = args.get("longitude")
            latitude = args.get("latitude")
            tolerance_or_radius = args.get("tolerance_or_radius")
            if longitude or latitude:
                if not tolerance_or_radius:
                    tolerance_or_radius = api_handler.toleranceGeoLocations_default
                    logger.info("%s radius set to default: " + str(api_handler.toleranceGeoLocations_default))

            uri = args.get("uri")
            threshold = args.get("threshold")
            transform_to_similarity = args.get("transform_to_similarity")
            if transform_to_similarity:
                transform_to_similarity = api_handler.evaluate_as_bool(transform_to_similarity)
            else:
                transform_to_similarity = False

            semantic_scores_between_targets = args.get("semantic_scores_between_targets")
            if semantic_scores_between_targets:
                semantic_scores_between_targets = api_handler.evaluate_as_bool(semantic_scores_between_targets)
            else:
                semantic_scores_between_targets = False

            response["semantic_distance_scores"] = []
            relations, semantic_scores = api_handler.retrieve_node_connectionsGeoFilter(uri, threshold,
                                                                                         user_input,
                                                                                         search_in_all_properties,
                                                                                         filtered_labels,
                                                                                         is_contemporary, filtered_properties,
                                                                                         filtered_takeAsPeriod, include_unknown_period,
                                                                                         filtered_startyear, filtered_endyear,
                                                                                         include_unknown_location, longitude,
                                                                                         latitude, tolerance_or_radius,
                                                                                         semantic_scores_between_targets,
                                                                                         transform_to_similarity)
            response["relations"] = relations
            if len(semantic_scores) == 1:
                response["semantic_distance_scores"] = semantic_scores[0]
            else:
                response["semantic_distance_scores"] = {}

        except Exception as e:
            logger.error("Could not retrieve connections of node: %s", e)
        return response


@ns_post.route("/getConnectionsCounterForNodeGeoFilter")
class ConnectionsCounterForNodeGeoFilter(Resource):
    @api.response(200, "Success", nodesCountInputModel)
    @ns_post.doc("getConnectionsCounterForNodeGeoFilter", parser=inputLabelsPropertiesGeoParserConnections)
    def post(self) -> OrderedDict:
        """
        Retrieve count of connected nodes for a given node uri
        : params: see example below
        : return: int connections count

        Example parameters (try other uris from "getMatchingUris" or "getAllUris"):
        {
            "userInput": "new",
            "labels":"Agent",
            "properties":"dbo:abstract",
            "include_search_in_all_properties": 1,
            "include_unknown_location": 0,
            "isContemporary": 1,
            "start_year": 1800,
            "end_year": 2000,
            "takeAsPeriod": 1,
            "include_unknown_period": 0,
            "latitude": 48.2,
            "longitude": 16.35,
            "tolerance_or_radius":10.0,
              "threshold": 10,
              "uri": "http://dbpedia.org/resource/United_States"
        }

        """
        response = OrderedDict()
        try:
            args = inputLabelsPropertiesGeoParserConnections.parse_args()
            user_input = args.get("userInput")
            if user_input:
                user_input = user_input.lower()

            search_in_all_properties = args.get("include_search_in_all_properties")
            if search_in_all_properties:
                search_in_all_properties = api_handler.evaluate_as_bool(search_in_all_properties)

            filtered_labels = args.get("labels")
            is_contemporary = args.get("isContemporary")

            filtered_properties = args.get("properties")

            filtered_takeAsPeriod = args.get("takeAsPeriod")

            filtered_startyear = args.get("start_year")
            filtered_endyear = args.get("end_year")
            if not filtered_takeAsPeriod and not filtered_takeAsPeriod==0:
                filtered_takeAsPeriod = set_default_takeAsPeriod(filtered_takeAsPeriod, filtered_startyear,
                                                                 filtered_endyear)

            include_unknown_period = args.get("include_unknown_period")
            if not include_unknown_period and not include_unknown_period == 0:
                include_unknown_period = api_handler.default_include_unknown

            include_unknown_location = args.get("include_unknown_location")
            if not include_unknown_location and not include_unknown_location == 0:
                include_unknown_location = api_handler.default_include_unknown

            longitude = args.get("longitude")
            latitude = args.get("latitude")
            tolerance_or_radius = args.get("tolerance_or_radius")
            if longitude or latitude:
                if not tolerance_or_radius:
                    tolerance_or_radius = api_handler.toleranceGeoLocations_default
                    logger.info("%s radius set to default: " + str(api_handler.toleranceGeoLocations_default))

            uri = args.get("uri")
            threshold = args.get("threshold")

            response["counter"] = api_handler.retrieve_counter_node_connectionsGeoFilter(uri, threshold,
                                                                                         user_input,
                                                                                         search_in_all_properties,
                                                                                         filtered_labels,
                                                                                         is_contemporary, filtered_properties,
                                                                                         filtered_takeAsPeriod, include_unknown_period,
                                                                                         filtered_startyear, filtered_endyear,
                                                                                         include_unknown_location, longitude,
                                                                                         latitude, tolerance_or_radius)

        except Exception as e:
            logger.error("Could not retrieve connections of node: %s", e)
        return response

@ns_post.route("/getAllNodes")
class AllNodes(Resource):
    @api.response(200, "Success", allNodesModel)
    @ns_post.doc("getAllNodes", parser=thresholdNodesParser)
    def post(self) -> OrderedDict:
        """
        Retrieve list of all nodes
        Without parameters the function returns the whole database in nodes
        : param thresholdNodes: integer
        : return: List of Nodes

        Example parameters (if parameter is not specified, the function returns all nodes -> that might take a while):
        {
        "thresholdNodes": 50
        }
        """
        response = OrderedDict()
        try:
            args = thresholdNodesParser.parse_args()
            nodes_threshold = args.get("thresholdNodes")
            #connections_threshold = args.get("thresholdConnections")
            #api_handler.logger.info("%s thresholds", nodes_threshold, connections_threshold)
            #return
            nodes = api_handler.retrieve_all_nodes(nodes_threshold)
            response["nodes"] = nodes
        except Exception as e:
            logger.error("Could not retrieve all nodes: %s", e)
        return response

@ns_post.route("/getAllNodesAndAllConnections")
class AllNodesAndConnections(Resource):
    @api.response(200, "Success", allNodesAndConnectionsModel)
    @ns_post.doc("getAllNodesAndAllConnections", parser=thresholdParser)
    def post(self) -> OrderedDict:
        """
        Retrieve list of all nodes and corresponding connections
        Without parameters the function returns the whole database in nodes and their respective relations
        : param thresholdNodes: integer
        : param thresholdConnections: integer
        : return: List of nodes

        Example parameters (get maximum for both values from "getStatistics"):
        {
        "thresholdNodes": 50,
        "thresholdConnections": 5
        }

        """
        response = OrderedDict()
        try:
            args = thresholdParser.parse_args()
            nodes_threshold = args.get("thresholdNodes")
            connections_threshold = args.get("thresholdConnections")
            #api_handler.logger.info("%s thresholds", nodes_threshold, connections_threshold)

            nodes = api_handler.retrieve_all_nodes(nodes_threshold)
            relations = []
            for n in nodes:
                relations.append(api_handler.retrieve_node_connections(n["uri"], [],
                                                                       connections_threshold,
                                                                       transform_to_similarity=False,
                                                                       retrieve_semantic_scores = False))
            response["nodes"] = nodes
            response["relations"] = relations
            response["semantic_distance_scores"] = []
        except Exception as e:
            logger.error("Could not retrieve all nodes: %s", e)
        return response


@ns_post.route("/getNodesAndConnectionsByLabels")
class NodesByLabel(Resource):
    @api.response(200, "Success", allNodesAndConnectionsModel)
    @ns_post.doc("getNodesAndConnectionsByLabels", parser=labelsThresholdParser)
    def post(self) -> OrderedDict:
        """
        Retrieve list of nodes and corresponding connections which match a specified set of labels within a threshold
        : param labels: string of Labels, separeted by ','
        : param threshold: integer
        : return: List of nodes

        Example parameters:
        {
        "labels": "Agent,Event,TopicalConcept",
        "threshold": 5,
        "semantic_scores_between_targets": 0,
        "transform_to_similarity":0
        }
        """
        response = OrderedDict()
        try:
            args = labelsThresholdParser.parse_args()
            labels = args.get("labels")
            threshold = args.get("threshold")
            nodes = api_handler.retrieve_nodes_by_labels(labels, threshold)
            relations = []
            transform_to_similarity = args.get("transform_to_similarity")
            if transform_to_similarity:
                transform_to_similarity = api_handler.evaluate_as_bool(transform_to_similarity)
            else:
                transform_to_similarity = False

            semantic_scores_between_targets = args.get("semantic_scores_between_targets")
            if semantic_scores_between_targets:
                semantic_scores_between_targets = api_handler.evaluate_as_bool(semantic_scores_between_targets)
            else:
                semantic_scores_between_targets = False
            semantic_scores_dict = {}

            for n in nodes:
                targets, semantic_scores_dict = api_handler.retrieve_node_connections(n["uri"], [],
                                                                                      threshold,
                                                                                      transform_to_similarity,
                                                                                      retrieve_semantic_scores = True,
                                                                                      semantic_scores_between_targets=semantic_scores_between_targets,
                                                                                      source_target_score_dict = semantic_scores_dict)
                relations.append(targets)
            response["nodes"] = nodes
            response["relations"] = relations
            response["semantic_distance_scores"] = []
            if semantic_scores_dict != {}:
                semantic_rel_obj = api_handler.get_semantic_rel_object(semantic_scores_dict, transform_to_similarity)
                response["semantic_distance_scores"] = semantic_rel_obj

        except Exception as e:
            logger.error("Could not retrieve nodes by labels: %s", e)
        return response


@ns_post.route("/getNodesCountByInputWithGeoFilter")
class NodesCountByInputWithGeoFilter(Resource):
    @api.response(200, "Success", nodesCountInputModel)
    @ns_post.doc("getNodesCountByInputWithGeoFilter", parser=inputLabelsPropertiesGeoParser)
    def post(self) -> OrderedDict:
        """
        Retrieve number of selected nodes for specified filters
        :param userInput: string
        :param labels: string of Labels, separeted by ','
        :param properties: string of Properties, separeted by ','
        :param search_in_all_properties: integer 0, 1 or null -> 1: search for user_input in all available properties of a node
        :param isContemporary: integer 0, 1 or null -> 0: only non-persons are included, 1: only persons are included,
                                                        null: every subcategory is included
        :param start_year: integer
        :param end_year: integer
        :param takeAsPeriod": integer 0, 1 or null -> 0: exact start and end year,
                                                      1: everything between start and end year
                                                     null: ignore param start and end year
        :param include_unknown_period: integer 0, 1 or null -> -> 1: unknown periods are included
        :param include_unknown_location": integer 0, 1 or null -> 1: unknown locations are included
        :param latitude: float
        :param longitude: float
        :param tolerance_or_radius: float , if latitude and longitude are specified radius is in km,
                                    else it is +/- the degree for the specified coordinate ; Default: 0.01
        :return: integer

        if you want to check, which location coordinates specify on the map, you can use
        https://nominatim.openstreetmap.org/ui/reverse.html?lat=22.95736474663196&lon=-101.93844595604823

        Example parameters
        {
          "userInput": "art",
          "labels":"Agent,Event",
          "properties":"dbo:abstract",
          "include_search_in_all_properties": 0,
          "isContemporary": 1,
          "start_year": 1800,
          "end_year": 1900,
          "takeAsPeriod": 1,
          "include_unknown_period": 0,
          "include_unknown_location": 0,
          "latitude": 54.6,
          "longitude": -2,
          "tolerance_or_radius": 10.0
        }
        """
        response = OrderedDict()
        try:
            args = inputLabelsPropertiesGeoParser.parse_args()
            user_input = args.get("userInput")
            if user_input:
                user_input = user_input.lower()

            search_in_all_properties = args.get("include_search_in_all_properties")
            if search_in_all_properties:
                search_in_all_properties = api_handler.evaluate_as_bool(search_in_all_properties)

            filtered_labels = args.get("labels")
            is_contemporary = args.get("isContemporary")

            filtered_properties = args.get("properties")

            filtered_takeAsPeriod = args.get("takeAsPeriod")

            include_unknown_period = args.get("include_unknown_period")
            #logger.info(include_unknown_period)
            if not include_unknown_period and not include_unknown_period == 0:
                include_unknown_period = api_handler.default_include_unknown
            #logger.info(include_unknown_period)

            filtered_startyear = args.get("start_year")
            filtered_endyear = args.get("end_year")
            if not filtered_takeAsPeriod and not filtered_takeAsPeriod==0:
                filtered_takeAsPeriod = set_default_takeAsPeriod(filtered_takeAsPeriod, filtered_startyear,
                                                                 filtered_endyear)

            include_unknown_location = args.get("include_unknown_location")
            if not include_unknown_location and not include_unknown_location == 0:
                include_unknown_location = api_handler.default_include_unknown
            longitude = args.get("longitude")
            latitude = args.get("latitude")
            tolerance_or_radius = args.get("tolerance_or_radius")
            if longitude or latitude:
                if not tolerance_or_radius:
                    tolerance_or_radius = api_handler.toleranceGeoLocations_default
                    logger.info("%s radius set to default: " + str(api_handler.toleranceGeoLocations_default))

            response["counter"] = api_handler.retrieve_counter_nodes_by_inputWithGeoFilter(user_input,
                                                                                           search_in_all_properties,
                                                                                           filtered_labels,
                                                                                           is_contemporary,
                                                                                           filtered_properties,
                                                                                           filtered_takeAsPeriod,
                                                                                           include_unknown_period,
                                                                                           filtered_startyear,
                                                                                           filtered_endyear,
                                                                                           include_unknown_location,
                                                                                           latitude, longitude,
                                                                                           tolerance_or_radius)
        except Exception as e:
            logger.error("Could not retrieve number of nodes by user input: %s", e)
        return response


@ns_post.route("/getNodesCountByInputWithGeoFilterByName")
class NodesCountByInputWithGeoFilterByName(Resource):
    @api.response(200, "Success", nodesCountInputModel)
    @ns_post.doc("getNodesCountByInputWithGeoFilterByName", parser=inputLabelsPropertiesGeoParserByName)
    def post(self) -> OrderedDict:
        """
        Retrieve number of selected nodes for specified filters
        :param userInput: string
        :param labels: string of Labels, separeted by ','
        :param properties: string of Properties, separeted by ','
        :param search_in_all_properties: integer 0, 1 or null -> 1: search for user_input in all available properties of a node
        :param isContemporary: integer 0, 1 or null -> 0: only non-persons are included, 1: only persons are included,
                                                        null: every subcategory is included
        :param start_year: integer
        :param end_year: integer
        :param takeAsPeriod": integer 0, 1 or null -> 0: exact start and end year,
                                                      1: everything between start and end year
                                                     null: ignore param start and end year
        :param include_unknown_period: integer 0, 1 or null -> -> 1: unknown periods are included
        :param include_unknown_location": integer 0, 1 or null -> 1: unknown locations are included
        :param continents: string of continents, separeted by ','
        :param countries: string of countries, separeted by ','
        :param tolerance_or_radius: float , if latitude and longitude are specified radius is in km,
                                    else it is +/- the degree for the specified coordinate
        :return: integer

        Attention: if continent and country is specified, the results are all nodes with coordinates
        in each whole continent and all nodes with coordinates in each country
        If you want to search for specific countries only, don't specify the continent parameter

        if you want to check, which location coordinates specify on the map, you can use
        https://nominatim.openstreetmap.org/ui/reverse.html?lat=22.95736474663196&lon=-101.93844595604823

        Example parameters:
        {
          "userInput": "art",
          "labels":"Agent,Event",
          "properties":"dbo:abstract",
          "include_search_in_all_properties": 1,
          "isContemporary": 1,
          "start_year": 1500,
          "end_year": 1900,
          "takeAsPeriod": 1,
          "include_unknown_period": 0,
          "include_unknown_location": 0,
          "continents": "america,europe",
          "countries": "germany"
        }
        """
        response = OrderedDict()
        try:
            args = inputLabelsPropertiesGeoParserByName.parse_args()
            user_input = args.get("userInput")
            if user_input:
                user_input = user_input.lower()

            search_in_all_properties = args.get("include_search_in_all_properties")
            if search_in_all_properties:
                search_in_all_properties = api_handler.evaluate_as_bool(search_in_all_properties)

            filtered_labels = args.get("labels")
            is_contemporary = args.get("isContemporary")

            filtered_properties = args.get("properties")

            filtered_takeAsPeriod = args.get("takeAsPeriod")

            filtered_startyear = args.get("start_year")
            filtered_endyear = args.get("end_year")
            if not filtered_takeAsPeriod and not filtered_takeAsPeriod==0:
                filtered_takeAsPeriod = set_default_takeAsPeriod(filtered_takeAsPeriod, filtered_startyear,
                                                                 filtered_endyear)

            include_unknown_period = args.get("include_unknown_period")
            if not include_unknown_period and not include_unknown_period == 0:
                include_unknown_period = api_handler.default_include_unknown

            include_unknown_location = args.get("include_unknown_location")
            if not include_unknown_location and not include_unknown_location == 0:
                include_unknown_location = api_handler.default_include_unknown

            continent = args.get("continents")
            country = args.get("countries")


            #filtered_labels = filtered_labels.split(" ")
            response["counter"] = api_handler.retrieve_counter_nodes_by_inputWithGeoFilterByName(user_input,
                                                                                                 search_in_all_properties,
                                                                                                 filtered_labels,
                                                                                                 is_contemporary,
                                                                                                 filtered_properties,
                                                                                                 filtered_takeAsPeriod,
                                                                                                 filtered_startyear,
                                                                                                 filtered_endyear,
                                                                                                 include_unknown_period,
                                                                                                 include_unknown_location,
                                                                                                 continent, country)
        except Exception as e:
            logger.error("Could not retrieve number of nodes by user input: %s", e)
        return response

#nodesByLabelModel -> to get all information of a node, not only description
@ns_post.route("/getNodesByInput")
class NodesByInput(Resource):
    @api.response(200, "Success", allNodesModel)#nodesInputModel
    @ns_post.doc("getNodesByInput", parser=inputLabelsPropertiesParser)
    def post(self) -> OrderedDict:
        """
         Retrieve selected nodes for specified filters
        :param userInput: string
        :param labels: string of Labels, separeted by ','
        :param properties: string of Properties, separeted by ','
        :param search_in_all_properties: integer 0, 1 or null
        :param isContemporary: integer 0, 1 or null
        :param start_year: integer
        :param end_year: integer
        :param takeAsPeriod": integer 0, 1 or null
        :param include_unknown_period: integer 0, 1 or null
        :return: List of nodes filtered by the parameters

        Example parameters
        {
          "userInput": "art",
          "labels":"Agent,Event",
          "properties":"dbo:abstract",
          "include_search_in_all_properties": 1,
          "isContemporary": 1,
          "start_year": 1800,
          "end_year": 1900,
          "takeAsPeriod": 1,
          "include_unknown_period": 0
        }

        """
        response = OrderedDict()
        try:
            args = inputLabelsPropertiesParser.parse_args()
            user_input = args.get("userInput")
            if user_input:
                user_input = user_input.lower()

            search_in_all_properties = args.get("include_search_in_all_properties")
            if search_in_all_properties:
                search_in_all_properties = api_handler.evaluate_as_bool(search_in_all_properties)

            filtered_labels = args.get("labels")
            is_contemporary = args.get("isContemporary")

            filtered_properties = args.get("properties")

            filtered_startyear = args.get("start_year")
            filtered_endyear = args.get("end_year")
            filtered_takeAsPeriod = args.get("takeAsPeriod")
            if not filtered_takeAsPeriod and not filtered_takeAsPeriod==0:
                filtered_takeAsPeriod = set_default_takeAsPeriod(filtered_takeAsPeriod, filtered_startyear,
                                                                 filtered_endyear)

            include_unknown_period = args.get("include_unknown_period")
            if not include_unknown_period and not include_unknown_period == 0:
                include_unknown_period = api_handler.default_include_unknown

            response["matchingNodes"] = api_handler.retrieve_nodes_by_input(user_input,
                                                                            search_in_all_properties,
                                                                            filtered_labels,
                                                                            is_contemporary,
                                                                            filtered_properties,
                                                                            filtered_takeAsPeriod,
                                                                            include_unknown_period,
                                                                            filtered_startyear,
                                                                            filtered_endyear)
        except Exception as e:
            logger.error("Could not retrieve list of nodes by user input: %s", e)
        return response


#nodesByLabelModel -> to get all information of a node, not only description
@ns_post.route("/getNodesByInputWithGeoFilterByName")
class NodesByInputWithGeoFilterByName(Resource):
    @api.response(200, "Success", allNodesModel)#nodesInputModel
    @ns_post.doc("getNodesByInputWithGeoFilterByName", parser=inputLabelsPropertiesGeoParserByName)
    def post(self) -> OrderedDict:
        """
        Retrieve selected nodes for specified filters
        :param userInput: string
        :param labels: string of Labels, separeted by ','
        :param properties: string of Properties, separeted by ','
        :param search_in_all_properties: integer 0, 1 or null
        :param isContemporary: integer 0, 1 or null
        :param start_year: integer
        :param end_year: integer
        :param takeAsPeriod": integer 0, 1 or null
        :param include_unknown_period: integer 0, 1 or null
        :param include_unknown_location": integer 0, 1 or null
        :param continents: string of continents, separeted by ','
        :param countries: string of countries, separeted by ','
        :param tolerance_or_radius: float , if latitude and longitude are specified radius is in km,
                                    else it is +/- the degree for the specified coordinate
        :return: List of nodes filtered by the parameters

        Attention: if continents and countries are specified, the results are all nodes with coordinates
        in each whole continent and all nodes with coordinates in each country
        If you want to search for specific countries only, don't specify the continent parameter

        if you want to check, which location coordinates specify on the map, you can use
        https://nominatim.openstreetmap.org/ui/reverse.html?lat=22.95736474663196&lon=-101.93844595604823

        Example parameters:
        {
          "userInput": "art",
          "labels":"Agent,Event",
          "properties":"dbo:abstract",
          "include_search_in_all_properties": 1,
          "isContemporary": 1,
          "start_year": 1800,
          "end_year": 1900,
          "takeAsPeriod": 1,
          "include_unknown_period": 0,
          "include_unknown_location": 0,
          "continents": "asia",
          "countries": "france"
        }
        """
        response = OrderedDict()
        try:
            args = inputLabelsPropertiesGeoParserByName.parse_args()
            user_input = args.get("userInput")
            if user_input:
                user_input = user_input.lower()

            search_in_all_properties = args.get("include_search_in_all_properties")
            if search_in_all_properties:
                search_in_all_properties = api_handler.evaluate_as_bool(search_in_all_properties)

            filtered_labels = args.get("labels")
            is_contemporary = args.get("isContemporary")

            filtered_properties = args.get("properties")

            filtered_startyear = args.get("start_year")
            filtered_endyear = args.get("end_year")
            filtered_takeAsPeriod = args.get("takeAsPeriod")
            if not filtered_takeAsPeriod and not filtered_takeAsPeriod==0:
                filtered_takeAsPeriod = set_default_takeAsPeriod(filtered_takeAsPeriod, filtered_startyear,
                                                                 filtered_endyear)

            include_unknown_period = args.get("include_unknown_period")
            if not include_unknown_period and not include_unknown_period == 0:
                include_unknown_period = api_handler.default_include_unknown

            include_unknown_location = args.get("include_unknown_location")
            if not include_unknown_location and not include_unknown_location == 0:
                include_unknown_location = api_handler.default_include_unknown
            continents = args.get("continents")
            countries = args.get("countries")

            response["matchingNodes"] = api_handler.retrieve_nodes_by_input_with_geoDataByName(user_input,
                                                                                        search_in_all_properties,
                                                                                        filtered_labels,
                                                                                        is_contemporary,
                                                                                        filtered_properties,
                                                                                        filtered_takeAsPeriod,
                                                                                        include_unknown_period,
                                                                                        filtered_startyear,
                                                                                        filtered_endyear,
                                                                                        include_unknown_location,
                                                                                         continents, countries)
        except Exception as e:
            logger.error("Could not retrieve list of nodes by user input: %s", e)
        return response


#nodesByLabelModel -> to get all information of a node, not only description
@ns_post.route("/getNodesByInputWithGeoFilter")
class NodesByInputWithGeoFilter(Resource):
    @api.response(200, "Success", allNodesModel)
    @ns_post.doc("getNodesByInputWithGeoFilter", parser=inputLabelsPropertiesGeoParser)
    def post(self) -> OrderedDict:
        """
        Retrieve selected nodes for specified filters
        :param userInput: string
        :param labels: string of Labels, separeted by ','
        :param properties: string of Properties, separeted by ','
        :param search_in_all_properties: integer 0, 1 or null -> 1: search for user_input in all available properties of a node
        :param isContemporary: integer 0, 1 or null -> 0: only non-persons are included, 1: only persons are included,
                                                        null: every subcategory is included
        :param start_year: integer
        :param end_year: integer
        :param takeAsPeriod": integer 0, 1 or null -> 0: exact start and end year,
                                                      1: everything between start and end year
                                                     null: ignore param start and end year
        :param include_unknown_period: integer 0, 1 or null -> -> 1: unknown periods are included
        :param include_unknown_location": integer 0, 1 or null -> 1: unknown locations are included
        :param latitude: float
        :param longitude: float
        :param tolerance_or_radius: float , if latitude and longitude are specified radius is in km,
                                    else it is +/- the degree for the specified coordinate
        :return: List of nodes filtered by the parameters

        if you want to check, which location coordinates specify on the map, you can use
        https://nominatim.openstreetmap.org/ui/reverse.html?lat=22.95736474663196&lon=-101.93844595604823

        Example parameters
        {
          "userInput": "art",
          "labels":"Agent",
          "properties":"dbo:abstract",
          "include_search_in_all_properties": 1,
          "isContemporary": 1,
          "start_year": 1800,
          "end_year": 1900,
          "takeAsPeriod": 1,
          "include_unknown_period": 0,
          "include_unknown_location": 0,
          "latitude": 54.6,
          "longitude": -2,
          "tolerance_or_radius": 10.0
        }
        """
        response = OrderedDict()
        try:
            args = inputLabelsPropertiesGeoParser.parse_args()
            user_input = args.get("userInput")
            if user_input:
                user_input = user_input.lower()

            search_in_all_properties = args.get("include_search_in_all_properties")
            if search_in_all_properties:
                search_in_all_properties = api_handler.evaluate_as_bool(search_in_all_properties)

            filtered_labels = args.get("labels")
            is_contemporary = args.get("isContemporary")

            filtered_properties = args.get("properties")

            filtered_startyear = args.get("start_year")
            filtered_endyear = args.get("end_year")
            filtered_takeAsPeriod = args.get("takeAsPeriod")
            #logger.info(filtered_takeAsPeriod)
            if not filtered_takeAsPeriod and not filtered_takeAsPeriod==0:
                filtered_takeAsPeriod = set_default_takeAsPeriod(filtered_takeAsPeriod, filtered_startyear,
                                                                 filtered_endyear)
            #logger.info(filtered_takeAsPeriod)
            include_unknown_period = args.get("include_unknown_period")
            #logger.info(include_unknown_period)
            if not include_unknown_period and not include_unknown_period==0:
                include_unknown_period = api_handler.default_include_unknown
            #logger.info(include_unknown_period)

            include_unknown_location = args.get("include_unknown_location")
            if not include_unknown_location and not include_unknown_location==0:
                include_unknown_location = api_handler.default_include_unknown
            longitude = args.get("longitude")
            latitude = args.get("latitude")
            tolerance_or_radius = args.get("tolerance_or_radius")
            if longitude or latitude:
                if not tolerance_or_radius:
                    tolerance_or_radius = api_handler.toleranceGeoLocations_default
                    logger.info("tolerance or radius set to default: " + str(api_handler.toleranceGeoLocations_default))

            response["matchingNodes"] = api_handler.retrieve_nodes_by_input_with_geoData(user_input,
                                                                            search_in_all_properties,
                                                                            filtered_labels,
                                                                            is_contemporary,
                                                                            filtered_properties,
                                                                            filtered_takeAsPeriod,
                                                                            include_unknown_period,
                                                                            filtered_startyear,
                                                                            filtered_endyear,
                                                                            include_unknown_location,
                                                                            latitude, longitude,
                                                                            tolerance_or_radius)
        except Exception as e:
            logger.error("Could not retrieve list of nodes by user input: %s", e)
        return response

#nodesByLabelModel -> to get all information of a node, not only description
@ns_post.route("/getNodesByUriWithGeoFilter")
class NearNodesByUriAndGeoFilter(Resource):
    @api.response(200, "Success", allNodesModel)
    @ns_post.doc("getNodesByUriWithGeoFilter", parser=uriRadiusParser)
    def post(self) -> OrderedDict:
        """
        Retrieve nearby located nodes for specified uri
        : param uri: string
        : param tolerance_or_radius: float (default radius is 0.01 km when radius is null)
        : return: List of nodes within specified radius or [] if coordinates are unknown for the specified uri

        Example Parameters
        {
          "uri": "http://dbpedia.org/resource/Mary_of_Waltham",
          "tolerance_or_radius": 1
        }
        """
        response = OrderedDict()
        try:
            args = uriRadiusParser.parse_args()

            uri = args.get("uri")
            tolerance_or_radius = args.get("tolerance_or_radius")
            if not tolerance_or_radius:
                tolerance_or_radius = api_handler.toleranceGeoLocations_default
                logger.info("radius set to default: " + str(api_handler.toleranceGeoLocations_default))

            response["NearByNodes"] = api_handler.retrieve_nodes_by_uri_and_radius(uri, tolerance_or_radius)
        except Exception as e:
            logger.error("Could not retrieve list of nodes by user input: %s", e)
        return response


@app.after_request
def add_header(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
    return response


