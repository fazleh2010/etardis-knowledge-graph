FROM node:lts-alpine3.14

# select working direction which will contain frontend application
WORKDIR /app

# copy application code
COPY ./ .
# clean up container
RUN rm -rf /app/node_modules

# add curl for healthcheck
RUN apk --no-cache add curl

# install react packages
RUN yarn install

# copy wait-for-api bash script
RUN chmod +x ./wait-for-api.sh

# expose port of react application
EXPOSE 3000

# health check
HEALTHCHECK CMD curl --fail http://etardis-frontend:3000 || exit 1

# entry point
CMD ["./wait-for-api.sh", "etardis-api:5000", "yarn", "start"]