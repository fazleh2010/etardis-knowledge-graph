FROM neo4j:4.3.6-community

# add apoc plugin
ENV APOC_VERSION 4.3.0.3
ENV APOC_URI https://github.com/neo4j-contrib/neo4j-apoc-procedures/releases/download/${APOC_VERSION}/apoc-${APOC_VERSION}-all.jar
RUN wget ${APOC_URI} && mv apoc-${APOC_VERSION}-all.jar ./plugins/apoc-${APOC_VERSION}-all.jar

# enable health check to verify if neo4j instance is accessible
HEALTHCHECK CMD curl --fail http://etardis-graph:7474 || exit 1

# entry point which starts neo4j instance
CMD ["neo4j"]

