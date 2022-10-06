# Docker Paperspace

Dockerfiles for my costum docker container used in Paperspace.

# Instructions

**1. Manually build and push the image**

```bash
# Login to DockerHub
docker login -u <username> -p <password>

# Build the image
docker build -t miguelcalado/docker-paperspace:latest .

# Test the build
docker run -it --name myapp --rm --volume --net=host miguelcalado/docker-paperspace:latest bash

# Deploy it to DockerHub
docker push miguelcalado/docker-paperspace:latest
```

**2. CD**

Commit and push to the repository, GitHub deals with the rest:

```bash
name: cd

on:
  push:
    branches:
      - 'main'

jobs:
  docker:
    runs-on: ubuntu-latest
    steps:
      -
        name: Set up QEMU
        uses: docker/setup-qemu-action@v2
      -
        name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v2
      -
        name: Login to DockerHub
        uses: docker/login-action@v2
        with:
          username: ${{ secrets.DOCKERHUB_USERNAME }}
          password: ${{ secrets.DOCKERHUB_TOKEN }}
      -
        name: Build and push
        uses: docker/build-push-action@v3
        with:
          push: true
          tags: <username>/<docker-image>:<tag>
```