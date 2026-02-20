#!/bin/bash
# ============================================================================
# EC2 Setup Script - Upload overcooked_ai-master to EC2
# ============================================================================
# This script:
# 1. Cleans up the EC2 instance (removes old files)
# 2. Uploads the overcooked_ai-master repository
# 3. Sets up the Python environment with required dependencies
#
# Environment variables (set these before running):
#   PEM_KEY   - Path to your AWS PEM key file
#   EC2_HOST  - EC2 instance hostname or IP address
#   EC2_USER  - EC2 user (default: ec2-user)
# ============================================================================

set -e  # Exit on error

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_REPO="$(dirname "$SCRIPT_DIR")"

# EC2 Configuration (use environment variables with defaults)
PEM_KEY="${PEM_KEY:-$HOME/.ssh/overcooked.pem}"
EC2_USER="${EC2_USER:-ec2-user}"
EC2_HOST="${EC2_HOST:-}"
EC2_DEST="/home/$EC2_USER"

# Validate required variables
if [ -z "$EC2_HOST" ]; then
    echo "Error: EC2_HOST is not set"
    echo "Usage: export EC2_HOST=<your-ec2-ip> && ./setup_ec2.sh"
    exit 1
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}============================================${NC}"
echo -e "${YELLOW}EC2 Setup Script - overcooked_ai-master${NC}"
echo -e "${YELLOW}============================================${NC}"

# Check if PEM key exists
if [ ! -f "$PEM_KEY" ]; then
    echo -e "${RED}Error: PEM key not found at $PEM_KEY${NC}"
    exit 1
fi

# Ensure correct permissions on PEM key
chmod 400 "$PEM_KEY"

echo -e "\n${GREEN}Step 1: Cleaning up EC2 instance...${NC}"
ssh -i "$PEM_KEY" -o StrictHostKeyChecking=no "$EC2_USER@$EC2_HOST" << 'ENDSSH'
    echo "Current disk usage:"
    df -h /
    
    echo ""
    echo "Removing old files..."
    
    # Remove old overcooked directories if they exist
    rm -rf ~/overcooked_ai-master 2>/dev/null || true
    rm -rf ~/overcooked-reproduction 2>/dev/null || true
    rm -rf ~/results 2>/dev/null || true
    rm -rf ~/test_results 2>/dev/null || true
    
    # Clean pip cache
    pip3 cache purge 2>/dev/null || pip cache purge 2>/dev/null || true
    
    # Clean conda cache if conda exists
    if command -v conda &> /dev/null; then
        conda clean --all -y 2>/dev/null || true
    fi
    
    echo ""
    echo "Disk usage after cleanup:"
    df -h /
    
    echo ""
    echo "EC2 cleanup complete!"
ENDSSH

echo -e "\n${GREEN}Step 2: Uploading overcooked_ai-master to EC2...${NC}"

# Create a temporary directory for the upload (excluding large/unnecessary files)
TEMP_DIR=$(mktemp -d)
echo "Creating clean copy in $TEMP_DIR..."

# Use rsync to copy, excluding unnecessary files
rsync -av --progress \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.pytest_cache' \
    --exclude='*.egg-info' \
    --exclude='build' \
    --exclude='dist' \
    --exclude='.ipynb_checkpoints' \
    --exclude='results' \
    --exclude='test_results' \
    --exclude='smoke_test' \
    --exclude='*.pkl' \
    --exclude='*.pickle' \
    --exclude='wandb' \
    "$LOCAL_REPO/" "$TEMP_DIR/overcooked_ai-master/"

# Upload to EC2
echo "Uploading to EC2..."
scp -i "$PEM_KEY" -o StrictHostKeyChecking=no -r \
    "$TEMP_DIR/overcooked_ai-master" \
    "$EC2_USER@$EC2_HOST:$EC2_DEST/"

# Clean up temp directory
rm -rf "$TEMP_DIR"

echo -e "\n${GREEN}Step 3: Setting up Python environment on EC2...${NC}"
ssh -i "$PEM_KEY" "$EC2_USER@$EC2_HOST" << 'ENDSSH'
    cd ~/overcooked_ai-master
    
    echo "Installing Python dependencies..."
    
    # Use python3/pip3 (standard on EC2/Amazon Linux)
    PYTHON=$(which python3 || which python)
    PIP=$(which pip3 || which pip)
    echo "Using Python: $PYTHON"
    echo "Using Pip: $PIP"
    $PYTHON --version
    
    # Upgrade pip
    $PIP install --upgrade pip
    
    # Install JAX dependencies
    $PIP install jax jaxlib flax optax
    
    # Install additional requirements if they exist
    if [ -f requirements.txt ]; then
        $PIP install -r requirements.txt
    fi
    
    # Install the package in development mode
    echo ""
    echo "Installing package in development mode..."
    $PIP install -e . -v
    
    # Add src to PYTHONPATH as a fallback (in case pip install -e doesn't work properly)
    export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
    echo "PYTHONPATH=$PYTHONPATH"
    
    # Also add to .bashrc for future sessions
    if ! grep -q "overcooked_ai-master/src" ~/.bashrc 2>/dev/null; then
        echo "" >> ~/.bashrc
        echo "# Overcooked AI PYTHONPATH" >> ~/.bashrc
        echo 'export PYTHONPATH="${PYTHONPATH}:${HOME}/overcooked_ai-master/src"' >> ~/.bashrc
        echo "Added PYTHONPATH to ~/.bashrc"
    fi
    
    echo ""
    echo "Verifying installation..."
    $PYTHON -c "
import sys
print(f'Python path: {sys.path[:3]}...')

import jax
import flax
import optax
print(f'JAX: {jax.__version__}')
print(f'Flax: {flax.__version__}')
print(f'Optax: {optax.__version__}')

# Check if overcooked_ai_py is installed
try:
    import overcooked_ai_py
    print(f'overcooked_ai_py: OK')
except ImportError as e:
    print(f'overcooked_ai_py: FAILED - {e}')

# Check if human_aware_rl is installed
try:
    import human_aware_rl
    print(f'human_aware_rl: OK')
except ImportError as e:
    print(f'human_aware_rl: FAILED - {e}')

# Check specific imports needed for training
from human_aware_rl.jaxmarl.ppo import PPOTrainer, PPOConfig
print('PPOTrainer imported successfully!')

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
print('OvercookedGridworld imported successfully!')
"
    
    echo ""
    echo "Setup complete!"
ENDSSH

echo -e "\n${GREEN}============================================${NC}"
echo -e "${GREEN}EC2 Setup Complete!${NC}"
echo -e "${GREEN}============================================${NC}"
echo -e "Repository uploaded to: ${EC2_USER}@${EC2_HOST}:${EC2_DEST}/overcooked_ai-master"
echo -e "\nTo SSH into the instance:"
echo -e "  ssh -i $PEM_KEY $EC2_USER@$EC2_HOST"
echo -e "\nTo run training:"
echo -e "  ./scripts/train_ec2.sh"

