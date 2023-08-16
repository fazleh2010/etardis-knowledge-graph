import ast
from collections import Counter, defaultdict
from typing import List, Dict

from py2neo import Graph

from src.crawling.graph_handler2 import GraphHandler


class GuessProperties:
    def __init__(self, url: str, username: str, password: str) -> None:
        try:
            self.graph = Graph(url, auth=(username, password))
        except Exception as e:
            raise RuntimeError(e)

    def simple_guess(self, property: str, threshold: int) -> Dict:
        to_update = defaultdict(List)

        unknown_nodes = []
        query = f"""
            MATCH (n) 
            WHERE n.{property} = "unknown" 
            RETURN n.uri AS uri
        """
        results = self.graph.run(query)
        for res in results:
            unknown_nodes.append(res.get("uri"))

        for uri in unknown_nodes:
            query = f"""
            MATCH (n)-[r]-(m) WHERE n.uri = "{uri}" AND m.{property} <> "unknown" RETURN m.{property} AS p
            """
            results = self.graph.run(query)

            if property == "locations":
                locations = []
                for res in results:
                    if isinstance(res.get("p"), list):
                        locations.append(res.get("p"))
                    else:
                        loc = ast.literal_eval(res.get("p"))
                        for l in loc:
                            locations.append(l)

                # only consider as predictable if count is greater than threshold
                common_list = Counter([tuple(x) for x in locations]).most_common()
                common_above_threshold = [list(tup[0]) for tup in common_list if tup[1] > threshold]
                if common_above_threshold:
                    to_update[uri] = common_above_threshold

            elif property == "period":
                period = []
                for res in results:
                    period.append(res.get("p"))
                # only consider most common date
                if period:
                    most_common, _ = Counter([tuple(x) for x in period]).most_common(1)[0]
                    to_update[uri] = list(most_common)

            else:
                print("Selected property is not supported yet.")

        return to_update


url = "bolt://localhost:7687"
username = "neo4j"
password = "password"
g = GuessProperties(url, username, password)

to_update = g.simple_guess("period", 0)
print(len(to_update))
# print(to_update)
