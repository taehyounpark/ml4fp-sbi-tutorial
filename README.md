# ML4FP SBI Tutorial

This repository houses the simulation-based inference tutorial to be presented at ML4FP 2025.

## Setup

First, build & launch the container by
```sh
apptainer build ml4fp-sbi-tutorial.sif container.def
apptainer shell --nv ml4fp-sbi-tutorial.sif
```

Inside the container, we will install our Python modules to be used in the tutorial:
```sh
pip install -e .
```

And now you're ready to go! Head over to notebooks in `tutorials/`