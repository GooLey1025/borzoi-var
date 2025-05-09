# Borzoi-var

A variant of the Borzoi project, focusing on genomic variant analysis and prediction. This repository contains modified versions of Borzoi and its dependencies (Baskerville and Westminster) with custom enhancements for variant analysis.

## Overview

Borzoi-var is a specialized version of the Borzoi deep learning model, adapted for genomic variant analysis. It combines the power of Borzoi's sequence modeling with custom modifications for improved variant effect prediction and analysis.

## Project Structure

```
Borzoi-var/
├── borzoi/          # Modified Borzoi package for variant analysis
├── baskerville/     # Enhanced Baskerville package
└── westminster/     # Customized Westminster package
```



## Setup

1. Clone the repository:
```bash
git clone https://github.com/your-username/borzoi-var.git
cd borzoi-var
```

2. Install the packages in editable mode:
```bash
pip install -e ./borzoi
pip install -e ./baskerville
pip install -e ./westminster
```

## Development

Each package is maintained as a separate git repository, allowing for independent version control while keeping everything organized in one place. The modifications focus on improving variant analysis capabilities while maintaining compatibility with the original packages.

## License

See individual package licenses in their respective directories.

## Acknowledgments

This project is based on the original Borzoi model and its dependencies, with custom modifications for variant analysis. 