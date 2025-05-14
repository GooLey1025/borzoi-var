# Borzoi-var

This repository is a customized variant of the Borzoi project, developed to support my research activities.

## Overview

Borzoi-var is a specialized version of the [Borzoi](https://github.com/calico/borzoi) deep learning model.

## Project Structure

```
Borzoi-var/
├── borzoi/          # Modified Borzoi package for variant analysis
├── baskerville/     # Enhanced Baskerville package
└── westminster/     # Customized Westminster package
```



## Setup

1. Create conda_env
```bash
conda env create -n borzoi_var python=3.10
```
2. Clone the repository:
```bash
git clone https://github.com/GooLey1025/borzoi-var.git
cd borzoi-var
```

3. Install the packages in editable mode:
```bash
pip install -e ./borzoi
pip install -e ./baskerville
pip install -e ./westminster
```
4. Manually set enviroment variables in this conda borzoi_var environment:
```bash
# You should modify your $CONDA_PREFIX,$USER_PATH,$USER.
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
cat <<EOF > $CONDA_PREFIX/etc/conda/activate.d/env_var.sh
export BORZOI_DIR=/home/$USER_PATH/Borzoi/borzoi
export PATH=\$BORZOI_DIR/src/scripts:\$PATH
export PYTHONPATH=\$BORZOI_DIR/src/scripts:\$PYTHONPATH

export BASKERVILLE_DIR=/home/$USER_PATH/Borzoi/baskerville
export PATH=\$BASKERVILLE_DIR/src/baskerville/scripts:\$PATH
export PYTHONPATH=\$BASKERVILLE_DIR/src/baskerville/scripts:\$PYTHONPATH

export WESTMINSTER_DIR=/home/$USER_PATH/Borzoi/westminster
export PATH=\$WESTMINSTER_DIR/src/westminster/scripts:\$PATH
export PYTHONPATH=\$WESTMINSTER_DIR/src/westminster/scripts:\$PYTHONPATH

export BORZOI_CONDA=/home/$USER/anaconda3/etc/profile.d/conda.sh
export BORZOI_HG38=\$BORZOI_DIR/examples/hg38
export BORZOI_MM10=\$BORZOI_DIR/examples/mm10
export BASKERVILLE_CONDA=\$BORZOI_CONDA
EOF
```


## Acknowledgments

This project is based on the original Borzoi model and its dependencies, with custom modifications for research activities. 