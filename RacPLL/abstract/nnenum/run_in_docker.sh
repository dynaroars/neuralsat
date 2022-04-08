#!/bin/bash
# run code in docker

PREFIX=nnenum

CONTAINER=${PREFIX}_container
IMAGE=${PREFIX}_image

rm -f ${RESULT_FILE}
echo "Running in Docker using container name $CONTAINER and image name $IMAGE"

docker kill $CONTAINER
docker stop $CONTAINER
docker rm $CONTAINER

docker build . -t $IMAGE

docker run -d --name $CONTAINER $IMAGE tail -f /dev/null

docker run $IMAGE
RV=$?
# "docker ps" should now list the image as running

# to get a shell, remove the lines at the end that delete the container and do: "docker exec -it $CONTAINER bash"

#docker cp $CONTAINER:/${RESULT_FILE} ${RESULT_FILE}

docker kill $CONTAINER
docker stop $CONTAINER
docker rm $CONTAINER

echo "Done with exit code $RV."

exit $RV
