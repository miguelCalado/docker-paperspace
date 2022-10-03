# Docker-paperspace
Dockerfiles for my costum docker container used in Paperspace

# Instructions

```bash
# Login to DockerHub
docker login -u <username> -p <password>

# Build the image
docker build -t miguelcalado/docker-paperspace:latest .

# Deploy it to DockerHub
docker push miguelcalado/docker-paperspace:latest
```