#!/bin/bash

# Exit on any error
set -e

echo "Setting up development environment..."

# Install git and other utilities
echo "Installing system packages..."
apt-get update && apt-get install -y rsync wget git zsh tmux htop curl && rm -rf /var/lib/apt/lists/*

# Update pip to the latest version
echo "Updating pip..."
pip install --no-cache-dir --upgrade pip

# Uninstall apex first
echo "Uninstalling apex..."
pip uninstall -y apex

# Upgrade pip again
python -m pip install --upgrade pip

# Install flash-attn
echo "Installing flash-attn..."
# pip install --no-cache-dir flash-attn==2.6.3

# Install oh-my-zsh
echo "Installing oh-my-zsh..."
sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended

# Install powerlevel10k theme
echo "Installing powerlevel10k theme..."
git clone --depth=1 https://github.com/romkatv/powerlevel10k.git ${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}/themes/powerlevel10k

# Install zsh-autosuggestions
echo "Installing zsh-autosuggestions..."
git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions

# Download zsh-syntax-highlighting
echo "Installing zsh-syntax-highlighting..."
git clone --depth=1 https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting

# Configure zsh plugins
echo "Configuring zsh plugins..."
sed -i 's/plugins=(git)/plugins=(git zsh-autosuggestions zsh-syntax-highlighting)/' ~/.zshrc

# Add zsh-completions to fpath
echo "Adding zsh-completions to fpath..."
echo 'fpath+=${ZSH_CUSTOM:-${ZSH:-~/.oh-my-zsh}/custom}/plugins/zsh-completions/src' >> ~/.zshrc

# Set git configuration
echo "Setting git configuration..."
git config --global user.name "Runyu Lu"

# Install pre-commit
echo "Installing pre-commit..."
uv tool install pre-commit

# Note: Uncomment the following line if you want to install pre-commit hooks
# pre-commit install

# put huggingface token (requires HF_TOKEN env var)
echo "Putting huggingface token..."
mkdir -p ~/.huggingface
if [ -z "$HF_TOKEN" ]; then
    echo "Warning: HF_TOKEN not set. Set it with: export HF_TOKEN=hf_your_token_here"
else
    echo "$HF_TOKEN" > ~/.huggingface/token
fi

# Install with uv
uv venv .venv --python 3.10
uv pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
uv pip install -e ".[diffusers]"
uv pip install --no-build-isolation flash-attn==2.6.3

echo "Environment setup complete!"
echo "To switch to zsh, run: exec zsh"
exec zsh
