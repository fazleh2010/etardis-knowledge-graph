#!/usr/bin/env bash

export FLASK_APP=graph_api.py FLASK_ENV=production FLASK_DEBUG=0
export NEO4J_URL=0.0.0.0 NEO4J_BOLT_PORT=7687 NEO4J_USER=neo4j NEO4J_PASSWD=password
gunicorn -w 4 -b 0.0.0.0:5000 graph_api:app