# eTaRDiS-knowledge-graph
[<img src="[https://github.com/fazleh2010/etardis-knowledge-graph/blob/main/eTarDiS.pdf](https://github.com/fazleh2010/etardis-knowledge-graph/blob/main/eTarDiS.pdf)" width="50%">](https://www.youtube.com/watch?v=okoJTP2MTDc&t=65s)

## eTaRDiS-knowledge-graph

This repository contains scripts to build and maintain the eTaRDiS knowledge graph based on Neo4j graph database.

## Graph database

The applied database is a Neo4j graph database. Example queries are listed in [Confluence](https://biedigital.atlassian.net/wiki/spaces/ET/pages/581959685/Neo4j+Beispiel+Queries).

### Existing neo4j database files

Different datasets can be applied by unpacking and replacing `$PATH/etardis-knowledge-graph/data/neo4j_data` folder.

-   data set is a union of all partial data sets: `$PATH/etardis-knowledge-graph/data/neo4j_data_bundled.tar.gz`
-   entry point [100 years war](http://dbpedia.org/resource/Hundred_Years'_War) with search depth 1: `$PATH/etardis-knowledge-graph/data/neo4j_data_100_years_war.tar.gz`
-   entry point [cornwall](http://dbpedia.org/resource/Cornwall) with search depth 1: `$PATH/etardis-knowledge-graph/data/neo4j_data_cornwall.tar.gz`
-   entry point [drag](<http://dbpedia.org/resource/Drag_(clothing)>) with search depth 1: `$PATH/etardis-knowledge-graph/data/neo4j_data_drag.tar.gz`
-   entry point [einstein](http://dbpedia.org/resource/Albert_Einstein) with search depth 1: `$PATH/etardis-knowledge-graph/data/neo4j_data_einstein.tar.gz`

## Crawling process

The crawling process is a standalone process and is executed separately.
All related scripts can be found in the `$PATH/etardis-knowledge-graph/src/crawling path`.

The process is initialized using the crawl_data.py script. Therefore the following variables have to be adjusted:

-   url = "bolt://localhost:7687"
-   username = "neo4j"
-   password = "password"
-   data_path = "$PATH/etardis-knowledge-graph/data"

In addition, the `search_depth` and the entrypoint `uri` of a dbpedia entry must be set in the main function.

Beside the actual crawling process, where the received data is stored in files in the `$PATH/etardis-knowledge-graph/data` folder, the `insert_into_neo4j()` function can also be used to load the data into the specified and accessible Neo4j instance.

## Development

### Build local Neo4j instance

1. Go to directory of neo4j dockerfile: `$PATH/etardis-knowledge-graph/src/crawling`
2. Handle user management for Neo4j. Due to use of community edition only available user is `neo4j` and can not be changed.
    1. Adjust password for authentication here: `NEO4J_AUTH: neo4j/password`
3. Build docker container: `docker build -f Dockerfile -t neo4j-etardis:latest .`

### Start local Neo4j instance

1. Install docker and docker-compose on your system
2. Unpack neo4j_data folder (e.g. `$PATH/etardis-knowledge-graph/data/neo4j_data_bundled.tar.gz`)
3. Adjust volume path of neo4j_data folder in `$PATH/etardis-knowledge-graph/docker-compose-graph.yml` (consider only first part: `$PATH/etardis-knowledge-graph/data/neo4j_data`)
4. Start neo4j by typing `docker-compose -f docker-compose-graph.yml up` (in same directory where docker-compose-graph.yml file is)
5. Explore local instance of neo4j: `http://localhost:7474` (username: neo4j, password: password)
6. Stop neo4j by typing `docker-compose -f docker-compose-graph.yml down`

### Manually start API

1. Install python3 on your system
2. Go to api directory: `$PATH/etardis-knowledge-graph/src/api`
3. Install required python packages (`pip install -r requirements.txt`). Using virtual python environment is recommended.
4. Execute bash script (`./$PATH/etardis-knowledge-graph/src/api/env.sh`) which starts REST API at localhost:5000
5. Documentation of endpoints is provided by swagger and is available under `localhost:5000`

### Manually start React frontend

1. Install yarn and npm on your system.
2. Go to frontend directory: `$PATH/etardis-knowledge-graph/src/frontend`
3. Install required yarn packages: `yarn`
4. Go to `$PATH/etardis-knowledge-graph/src/frontend/package.json` file and adjust api path of proxy for local usage to: `"proxy": "http://localhost:5000"`
5. Start frontend (`yarn start`) which is accessible in any browser at `localhost:3000`

## Deployment

Using docker-compose, the individual components are executed together as Docker containers in a network.
Necessary configurations can be made in the respective docker-compose file which can be found at `$PATH/etardis-knowledge-graph`.
It also requires one of the provided Neo4j databases (e.g. `$PATH/etardis-knowledge-graph/data/neo4j_data_bundled.tar.gz`), which are mounted into the respective container using VOLUME.
To do this, the corresponding files must first be unpacked and the resulting folder must then be renamed to "neo4j_data".

### Development

`docker-compose -f docker-compose.yml down && docker-compose -f docker-compose.yml up --build`

After applying this command there are multiple services accessible under:

-   Neo4j database: `http://localhost:7474`
-   REST API: `http://localhost:5000`
-   React frontend: `http://localhost:3000`

### Production

`docker-compose -f docker-compose.prod.yml down && docker-compose -f docker-compose.prod.yml up --build`

After applying this command there are multiple services accessible under:

-   Neo4j database: `http://localhost:7474`
-   REST API: `http://localhost:5000`
-   React frontend: `http://localhost:80`

## TODO

-   update crawling process by considering all data from dbpedia and wikidata for each entry
-   update crawling process and API by only applying any name/uri formatting (such as dbo:, ...) at API step
-   update description entries according to [Confluence](https://biedigital.atlassian.net/wiki/spaces/ET/pages/615940108/Steckbriefinformationen)
-   update endpoints for VR usage (add boolean to skip nodes which contain any "unknown" property values)
-   update [helping texts and add colors](https://biedigital.atlassian.net/wiki/spaces/ET/pages/620822529/nderungsw+nsche+-+Hands-on+Workshop+am+15.03.2022) to distinguish between dbpedia and wikidata properties

## Further documentation

[Official Neo4j documentation](https://neo4j.com/docs/operations-manual/current/docker/introduction/)

[Flask](https://flask.palletsprojects.com/en/2.1.x/)

[Swagger flask-restx](https://flask-restx.readthedocs.io/en/latest/index.htmlflask)

[React](https://reactjs.org/docs/getting-started.html)

[Antd](https://ant.design/components/overview/)

[Anychart](https://docs.anychart.com/Quick_Start/Quick_Start)

[Docker](https://docs.docker.com/)
