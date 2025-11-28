#!/bin/bash
set -e

# Upgrade pip and install build tools
pip install --upgrade pip setuptools wheel

# Install tokenizers first with binary preference
# Use an older, stable version that definitely has wheels
pip install --only-binary=:all: tokenizers==0.13.3 || pip install tokenizers==0.13.3

# Install remaining requirements
pip install -r requirements.txt

