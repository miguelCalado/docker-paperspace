<img align="left" width="165" height="165" src="https://i.imgur.com/SWwQJRe.png" title="PaperSpace logo">

#
# Docker Paperspace
#
#


**Summary**: Dockerfile for my custom Docker image used in [Paperspace Gradient Notebooks](http://paperspace.com/).
It includes my personal setup and other ML/DL libraries (see [Setup-Info](Setup-Info.md) for more details).
The Docker image is hosted on [Dockerhub](https://hub.docker.com/repository/docker/miguelcalado/docker-paperspace).

**Issue**: Notebook instances deletes every environment variable created in the current session upon restarting. This can be troublesome whenever you've installed a new python package, set up cloud or Git credentials, or added a new nbextension to jupyter notebook/lab, requiring to repeat the same process and reinstall all the changes you made in the next session.

**Solution**: Create a Docker image, push it to a registry (e.g. [DockerHub](hub.docker.com)) and link it to your Notebook instance when creating. Changes that you want to be permanent, for instance, adding ```pandas``` to the python environment, you'll just need to modify the Dockerfile, build the image and upload it to the registry. In the next Notebook instantiation, Paperspace will pull the newest image available in your registry repository.

# Setup

**1. Manually build and push the image**

```bash
# Login to DockerHub
docker login -u <username> -p <password>

# Build the image
docker build -t miguelcalado/docker-paperspace:latest .

# Test the build
docker run -it --name paperspace_container --rm --volume --net=host miguelcalado/docker-paperspace:latest bash

# Deploy it to DockerHub
docker push miguelcalado/docker-paperspace:latest
```

**2. Countinous Delivery (Github Actions)**

Push your changes to the repository, GitHub deals with the rest:

```bash
name: cd

on:
  push:
    paths:
      - 'notebook.json'
      - 'Dockerfile'

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

Lastly, associate the Docker image pushed to a registry with your to be created Notebook. Go to advanced settings and add the name of the repository to the **Container name**.

<p align="center">
  <img width="666" src="https://i.imgur.com/scY3R7H.png">
</p>
