#!/bin/bash
set -e

# Upgrade pip and install build tools
pip install --upgrade pip setuptools wheel

# Try to install tokenizers with binary wheels first
pip install tokenizers==0.15.2 --only-binary=:all: || {
    echo "Binary wheels not available, installing Rust..."
    # Install Rust if needed (this might not work on Render, but worth trying)
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    export PATH="$HOME/.cargo/bin:$PATH"
    pip install tokenizers==0.15.2
}

# Install remaining requirements
pip install -r requirements.txt

