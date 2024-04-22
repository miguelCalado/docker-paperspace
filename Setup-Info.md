# System Information

This Docker image is design to run DL and DS experiments/training on whatever hardware specifications (meaning that it works on small, big or multiple GPUs), and is able to work locally, on cloud (e.g. [GCP](cloud.google.com)) or on specialized computing platforms (e.g. [Paperspace](https://www.paperspace.com/) and [JarvisLabs](https://cloud.jarvislabs.ai/)). It runs on **Ubuntu 20.04** and contains common packages (TensorFlow, PyTorch, Jax, HuggingFace, etc.) that work interchangebly and without conflict.

The image can be pulled from [DockerHub](https://hub.docker.com/repository/docker/miguelcalado/docker-paperspace), and be pulled as:

`docker pull miguelcalado/docker-paperspace:latest`

**References**: [Dockerfile best pratices](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/)
# Tables

The full specications are available in the following tables:

- [GPU drivers + CUDA](#place1)
- [Python packages](#Python-packages)
- [Deep learning frameworks](#Deep-learning-frameworks)
- [Jupyter Lab/Notebook](#place2)
    - [Extensions](#Extensions)
    - [Macros](#Macros)

## GPU drivers + CUDA<span id="place1"></span>

| **Software** | **Type** | **Version** |
|:---:|:---:|:---:|
| NVidia Driver | GPU | 510.73.05 |
| CUDA | GPU | 11.6.2 |
| CUDA toolkit | GPU | 11.6.2 |
| cuDNN | GPU | 8.4.1.*-1+cuda11.6 |

## Python packages

| **Python Package** | **Version** | **Install method** |
|:---:|:---:|:---:|
| numpy | 1.23.1 | pip3 |
| scipy | 1.8.1 | pip3 |
| pandas | 1.4.3 | pip3 |
| cloudpickle | 2.1.0 | pip3 |
| scikit-image | 0.19.3 | pip3 |
| scikit-learn | 1.1.1 | pip3 |
| matplotlib | 3.5.2 | pip3 |
| ipython | 8.4.0 | pip3 |
| ipykernel | 6.15.1 | pip3 |
| ipywidgets | 7.7.1 | pip3 |
| Cython | 0.29.30 | pip3 |
| tqdm | 4.64.0 | pip3 |
| gdown | 4.5.1 | pip3 |
| xgboost | 1.6.1 | pip3 |
| pillow | latest | pip3 |
| seaborn | 0.12.0 | pip3 |
| SQLAlchemy | 1.4.39 | pip3 |
| spacy | 3.4.0 | pip3 |
| nltk | 3.7 | pip3 |
| jsonify | 0.5 | pip3 |
| boto3 | 1.24.27 | pip3 |
| transformers | 4.22.1 | pip3 |
| sentence-transformers | 2.2.2 | pip3 |
| datasets | 2.3.2 | pip3 |
| opencv-python | 4.6.0.66 | pip3 |
| msal | latest | pip3 |
| elementpath | latest | pip3 |
| lxml | 4.9.1 | pip3 |
| wandb | latest | pip3 |
| types-requests | latest| pip3 |
| pytest | latest | pip3 |
| isort | latest | pip3 |
| black | latest | pip3 |
| flake8 | latest | pip3 |
| mypy | latest | pip3 |
| pyopenssl | latest | pip3 |
| cdifflib | latest | pip3 |
| nbqa | latest | pip3 |
| colour | latest | pip3 |
| pycocotools | latest | pip3 |
| tensorflow_datasets | latest | pip3 |

For the full python packages consult [dependencies.rst](dependencies.rst) - it was generated as ```pip freeze > dependencies.rst```.
## Deep learning frameworks

| **Python Package** | **Version** | **Install method** |
|:---:|:---:|:---:|
| Tensorflow | 2.11.0 | pip3 |
| torch | 1.12.1 | pip3 |
| torchvision | 0.13.1 | pip3 |
| torchaudio | 0.12.1 | pip3 |
| Jax | 0.3.23 | pip3 |
| Flax | 0.6.3 | pip3 |
| Transformers| 4.21.3 | pip3 |
| Datasets | 2.4.0 | pip3 |

## Jupyter Lab/Notebook <span id="place2"></span>

| **Python Package** | **Version** | **Install method** |
|:---:|:---:|:---:|
| jupyter | 1.0.0 | pip3 |
| jupyterlab | 3.4.6 | pip3 |
| notebook | 6.4.12 | pip3 |
| nodejs | 16.x latest | apt |
| jupyter_contrib_nbextensions | 0.5.1 | pip3 |
| jupyter_nbextensions_configurator | 0.4.1 | pip3 |

### Extensions

| **Software** | **Version** | **Type** | **Install method** | **Notes** |
|:---:|:---:|:---:|:---:|:---:|
| spellchecker | latest | Notebook extension | jupyter nbextension | Highlights incorrectly spelled words in Markdown and Raw cells |
| snippets_menu | latest | Notebook extension | jupyter nbextension | Menu of code snippets -> **TODO**: Add Stable Diffusion |
| snippets | latest | Notebook extension | jupyter nbextension | Costumizable code snippets. To add a new one you need to create a `snippets.json` file and `mv snippets.json $(jupyter --data-dir)/nbextensions/snippets/snippets.json` |
| freeze | latest | Notebook extension | jupyter nbextension | Freeze/block notebook cells |
| livemdpreview | latest | Notebook extension | jupyter nbextension | Preview markdown cells |
| highlight_selected_word | latest | Notebook extension | jupyter nbextension | Highlights all instances of the selected word in either the current cell's editor, or in all cells in the notebook |
| ExecuteTime | latest | Notebook extension | jupyter nbextension | Display when each cell has been executed and how long it took |
| toc2 | latest | Notebook extension | jupyter nbextension | Displays a table of content based on cell headers |
| jupyter_resource_usage | latest | Notebook extension | jupyter nbextension | Displays notebook memory usage + CPU and GPU utilization|
| jupyter black | latest | Notebook extension | jupyter nbextension | Button and macro that formats the code cells with `Black`|

### Macros

You can save your macros in `notebook.json` and save them under the directory `~/.jupyter/nbconfig/`.

My macros are listed below and you can check for more in [notebook.json](notebook.json).

```json
{
  "keys": {
    "command": {
      "bind": {
        "Ctrl-Shift-R": "jupyter-notebook:restart-kernel-and-clear-output",
        "Ctrl-Shift-Z": "jupyter-notebook:run-all-cells-above",
        "Ctrl-Shift-B": "jupyter-notebook:run-all-cells-below"
      }
    },
    "edit": {
      "bind": {
        "Ctrl-Shift-R": "jupyter-notebook:restart-kernel-and-clear-output",
        "Ctrl-Shift-Z": "jupyter-notebook:run-all-cells-above",
        "Ctr;-Shift-B": "jupyter-notebook:run-all-cells-below"
      }
    }
  }
}
```
