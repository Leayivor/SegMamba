#!/bin/bash

# Install causal-conv1d
cd causal-conv1d
python setup.py install
cd ..

# Install mamba
cd mamba
python setup.py install
cd ..
