#!/bin/bash
# ==============================================================================
# EC2 Setup Script - Run this on the EC2 instance after deployment
# ==============================================================================

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Overcooked RL EC2 Setup${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

cd ~/overcooked/human_aware_rl

echo -e "${YELLOW}Building Docker image...${NC}"
echo "This will take 10-15 minutes..."
echo ""

sudo docker build -t overcooked-rl -f Dockerfile.ec2 .

echo ""
echo -e "${GREEN}✓ Docker image built successfully!${NC}"
echo ""

echo -e "${YELLOW}Starting container...${NC}"

# Remove existing container if present
sudo docker rm -f overcooked 2>/dev/null || true

# Start container in background
sudo docker run -d --name overcooked overcooked-rl tail -f /dev/null

echo -e "${GREEN}✓ Container started!${NC}"
echo ""

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Setup Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "To enter the container:"
echo -e "  ${GREEN}sudo docker exec -it overcooked bash${NC}"
echo ""
echo "Quick experiment commands (run inside container):"
echo "  BC:     python experiments/bc_experiments.py"
echo "  PPO-SP: bash experiments/ppo_sp_experiments.sh"
echo "  PBT:    bash experiments/pbt_experiments.sh"
echo ""






