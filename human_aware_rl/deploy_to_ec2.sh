#!/bin/bash
# ==============================================================================
# Deploy Overcooked Human-Aware RL to EC2
# ==============================================================================
# This script:
# 1. Clears the EC2 instance home directory
# 2. Copies the codebase to the EC2 instance
# 3. Sets up Docker on the instance
# ==============================================================================

set -e

# Configuration
PEM_FILE="/Users/theochen/Downloads/overcooked-dpail.pem"
EC2_USER="ec2-user"
EC2_HOST="98.86.229.193"
REMOTE_DIR="/home/ec2-user/overcooked"
LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Overcooked RL EC2 Deployment Script${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Check if PEM file exists
if [ ! -f "$PEM_FILE" ]; then
    echo -e "${RED}Error: PEM file not found at $PEM_FILE${NC}"
    exit 1
fi

# Ensure correct permissions on PEM file
chmod 400 "$PEM_FILE"

echo -e "${YELLOW}Step 1: Cleaning EC2 instance...${NC}"
ssh -i "$PEM_FILE" -o StrictHostKeyChecking=no "$EC2_USER@$EC2_HOST" << 'ENDSSH'
    echo "Stopping any running Docker containers..."
    sudo docker stop $(sudo docker ps -aq) 2>/dev/null || true
    sudo docker rm $(sudo docker ps -aq) 2>/dev/null || true
    
    echo "Removing old project files..."
    rm -rf ~/overcooked
    rm -rf ~/human_aware_rl
    
    echo "Cleanup complete!"
ENDSSH

echo -e "${GREEN}✓ EC2 instance cleaned${NC}"
echo ""

echo -e "${YELLOW}Step 2: Preparing local files for transfer...${NC}"

# Create a temporary directory for the transfer
TEMP_DIR=$(mktemp -d)
echo "Using temp directory: $TEMP_DIR"

# Copy files, excluding unnecessary ones
rsync -a --progress \
    --exclude='.git' \
    --exclude='*.pyc' \
    --exclude='__pycache__' \
    --exclude='.DS_Store' \
    --exclude='*.egg-info' \
    --exclude='.idea' \
    --exclude='.vscode' \
    --exclude='node_modules' \
    --exclude='*.log' \
    "$LOCAL_DIR/" "$TEMP_DIR/human_aware_rl/"

echo -e "${GREEN}✓ Files prepared${NC}"
echo ""

echo -e "${YELLOW}Step 3: Copying codebase to EC2...${NC}"

# Create remote directory
ssh -i "$PEM_FILE" "$EC2_USER@$EC2_HOST" "mkdir -p $REMOTE_DIR"

# Copy files to EC2
scp -i "$PEM_FILE" -r "$TEMP_DIR/human_aware_rl" "$EC2_USER@$EC2_HOST:$REMOTE_DIR/"

# Clean up temp directory
rm -rf "$TEMP_DIR"

echo -e "${GREEN}✓ Codebase copied to EC2${NC}"
echo ""

echo -e "${YELLOW}Step 4: Setting up Docker on EC2...${NC}"

ssh -i "$PEM_FILE" "$EC2_USER@$EC2_HOST" << 'ENDSSH'
    # Install Docker if not present
    if ! command -v docker &> /dev/null; then
        echo "Installing Docker..."
        sudo yum update -y
        sudo yum install -y docker
        sudo systemctl start docker
        sudo systemctl enable docker
        sudo usermod -aG docker ec2-user
        echo "Docker installed!"
    else
        echo "Docker already installed"
        sudo systemctl start docker
    fi
    
    # Show Docker version
    docker --version
ENDSSH

echo -e "${GREEN}✓ Docker setup complete${NC}"
echo ""

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Deployment Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${YELLOW}Next Steps:${NC}"
echo ""
echo "1. SSH into the EC2 instance:"
echo -e "   ${GREEN}ssh -i \"$PEM_FILE\" $EC2_USER@$EC2_HOST${NC}"
echo ""
echo "2. Build the Docker image:"
echo -e "   ${GREEN}cd ~/overcooked/human_aware_rl${NC}"
echo -e "   ${GREEN}sudo docker build -t overcooked-rl -f Dockerfile.ec2 .${NC}"
echo ""
echo "3. Run the Docker container (interactive):"
echo -e "   ${GREEN}sudo docker run -it --name overcooked overcooked-rl${NC}"
echo ""
echo "4. Run experiments inside the container:"
echo "   - BC experiments:     python experiments/bc_experiments.py"
echo "   - PPO SP experiments: bash experiments/ppo_sp_experiments.sh"
echo "   - PBT experiments:    bash experiments/pbt_experiments.sh"
echo "   - PPO BC experiments: bash experiments/ppo_bc_experiments.sh"
echo "   - PPO HM experiments: bash experiments/ppo_hm_experiments.sh"
echo ""
echo "5. For long-running experiments, use screen or nohup:"
echo -e "   ${GREEN}sudo docker run -d --name overcooked-exp overcooked-rl bash -c 'cd experiments && bash ppo_sp_experiments.sh'${NC}"
echo ""
echo "6. To monitor running containers:"
echo -e "   ${GREEN}sudo docker logs -f overcooked-exp${NC}"
echo ""

