#!/bin/sh

set -e

HOST="$1"
# shift to ignore host parameter for execution
shift

until curl -s -I http://"$HOST" | grep -q "200 OK"; do
  echo "Wait until API is accessible at http://$HOST"
  sleep 5;
done

# execute further arguments
exec "$@"