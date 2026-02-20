#!/bin/bash
# ============================================================================
# Download Results from EC2
# ============================================================================
#
# Usage:
#   ./download_ec2_results.sh
#
# Environment variables (set these before running):
#   PEM_KEY   - Path to your AWS PEM key file
#   EC2_HOST  - EC2 instance hostname or IP address
#   EC2_USER  - EC2 user (default: ec2-user)
#
# Example:
#   export PEM_KEY="~/.ssh/my-key.pem"
#   export EC2_HOST="1.2.3.4"
#   ./download_ec2_results.sh

set -e

# Configuration (override with environment variables)
PEM_KEY="${PEM_KEY:-$HOME/.ssh/overcooked.pem}"
EC2_USER="${EC2_USER:-ec2-user}"
EC2_HOST="${EC2_HOST:-}"

# Validate required variables
if [ -z "$EC2_HOST" ]; then
    echo "Error: EC2_HOST is not set"
    echo "Usage: export EC2_HOST=<your-ec2-ip> && ./download_ec2_results.sh"
    exit 1
fi

if [ ! -f "$PEM_KEY" ]; then
    echo "Error: PEM key not found at $PEM_KEY"
    echo "Set PEM_KEY environment variable to the path of your key file"
    exit 1
fi

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_DEST="$(dirname "$SCRIPT_DIR")"

echo "Downloading paper reproduction results from EC2..."
echo "  EC2: $EC2_USER@$EC2_HOST"
echo "  Local: $LOCAL_DEST"
echo ""

mkdir -p "$LOCAL_DEST/results"
mkdir -p "$LOCAL_DEST/logs"

# Download paper reproduction results
echo "Downloading results/paper_reproduction..."
scp -i "$PEM_KEY" -r \
    "$EC2_USER@$EC2_HOST:~/overcooked_ai-master/results/paper_reproduction" \
    "$LOCAL_DEST/results/"

# Download logs
echo "Downloading logs/paper_reproduction..."
scp -i "$PEM_KEY" -r \
    "$EC2_USER@$EC2_HOST:~/overcooked_ai-master/logs/paper_reproduction" \
    "$LOCAL_DEST/logs/" 2>/dev/null || echo "  (No logs/paper_reproduction folder found)"

# Download training log
echo "Downloading training.log..."
scp -i "$PEM_KEY" \
    "$EC2_USER@$EC2_HOST:~/training.log" \
    "$LOCAL_DEST/logs/training_paper_reproduction.log" 2>/dev/null || echo "  (No training.log found)"

echo ""
echo "============================================"
echo "Download complete!"
echo "============================================"
echo "Results: $LOCAL_DEST/results/paper_reproduction/"
echo ""
