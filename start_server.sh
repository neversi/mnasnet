#!/bin/bash

docker rm -f $(docker ps -a -q)
docker buildx build -t mnasnet . 
docker run -d --name mycontainer -p 80:80 mnasnet 
