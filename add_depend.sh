#!/bin/bash

# update
sudo apt-get update

# Install dependent packages and automatically confirm all installations
sudo apt-get install -y build-essential
sudo apt-get install -y pkg-config
sudo apt-get install -y gfortran
sudo apt-get install -y clang
sudo apt-get install -y libopenblas-dev
sudo apt-get install -y llvm
sudo apt-get install -y google-perftools