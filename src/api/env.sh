#!/usr/bin/env bash

export FLASK_APP=graph_api.py FLASK_ENV=development FLASK_DEBUG=1
export NEO4J_URL=0.0.0.0 NEO4J_BOLT_PORT=7687 NEO4J_USER=neo4j NEO4J_PASSWD=password
flask run --host=localhost --port=5000
