FROM python:3.9.6

# environment variables
ENV FLASK_APP graph_api.py
ENV FLASK_ENV development
ENV FLASK_DEBUG 1

ENV NEO4J_URL 0.0.0.0
ENV NEO4J_BOLT_PORT 7687
ENV NEO4J_USER neo4j
ENV NEO4J_PASSWD password

WORKDIR /app/graph

# install required python packaged
RUN apt-get update
RUN apt-get install -y libgdal-dev
COPY ./requirements.txt .
RUN pip install -r ./requirements.txt

RUN python -m spacy download en_core_web_sm

# copy flask api
COPY ./.flaskenv .
COPY ./graph_api.py .
COPY ./geo_data ./geo_data
COPY ./fine_tuned_model_all-MiniLM-L6-v2 ./fine_tuned_model_all-MiniLM-L6-v2

# copy wait-for-neo4j bash script
COPY ./wait-for-neo4j.sh .
RUN chmod +x ./wait-for-neo4j.sh

# expose port of flask application
EXPOSE 5000

HEALTHCHECK CMD curl --fail http://etardis-api:5000 || exit 1

# entry point
CMD ["./wait-for-neo4j.sh", "etardis-graph:7474", "flask", "run", "--host=etardis-api", "--port=5000"]
