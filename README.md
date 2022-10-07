# Docker Paperspace

Dockerfiles for a custom docker container used in [Paperspace](http://paperspace.com/).

**Issue**: Notebook instances deletes every environment variable created in the current session upon restarting. This can be troublesome whenever you've installed a new python package, set up cloud or Git credentials, or added a new nbextension to jupyter notebook/lab, requiring to repeat the same process and reinstall all the changes you made in the next session.

**Solution**: Create a Docker image, push it to a registry (e.g. DockerHub) and add it to your Notebook instance when creating. Changes that you want to be permanent, for instance, adding ```pandas``` to the python environment, you'll just need to modify the Dockerfile, build the image and upload it to the registry. In the next Notebook instantiation, Paperspace will pull the newest image available in your registry repository.

# Setup

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

Lastly, associate the Docker Image pushed to a registry with your to be created Notebook. Go to advanced settings and add the name of the repository to the **Container name**.

<p align="center">
  <img width="666" src="https://i.imgur.com/scY3R7H.png">
</p>