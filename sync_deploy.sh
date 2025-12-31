#!/bin/bash

# Remote configuration
REMOTE_USER="weizixiang"
REMOTE_HOST="210.75.240.33"
REMOTE_PROJECT_PATH="/home/weizixiang/dev/wk/github/pymetric-1-lego"

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Sync files
echo "Syncing project from ${SCRIPT_DIR} to ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PROJECT_PATH}..."

# Ensure remote directory exists
# ssh ${REMOTE_USER}@${REMOTE_HOST} "mkdir -p ${REMOTE_PROJECT_PATH}"

# Use rsync for efficient syncing
# -a: archive mode (recursive, preserves permissions, times, etc.)
# -v: verbose output
# -z: compress file data during the transfer
# -P: same as --partial --progress
# --delete: delete extraneous files from dest dirs (optional, be careful)
rsync -avzP \
    --exclude='.git/' \
    --exclude='.gitignore' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='*.pyo' \
    --exclude='.DS_Store' \
    --exclude='.idea/' \
    --exclude='.vscode/' \
    --exclude='venv/' \
    --exclude='.venv/' \
    --exclude='*.egg-info/' \
    --exclude='build/' \
    --exclude='dist/' \
    "${SCRIPT_DIR}/" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PROJECT_PATH}/"

if [ $? -eq 0 ]; then
    echo "Sync completed successfully."
else
    echo "Error: Sync failed."
    exit 1
fi
