#!/bin/bash

VAST_HOST="70.69.205.56"
VAST_PORT="57258"
WORKSPACE="/workspace/casino-of-life"

# Sync package files
rsync -avz -e "ssh -p $VAST_PORT" \
    --exclude 'venv' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    ./ root@$VAST_HOST:$WORKSPACE/

# Install requirements
ssh -p $VAST_PORT root@$VAST_HOST "cd $WORKSPACE && source venv/bin/activate && pip install -r requirements.txt"

# Restart the service
ssh -p $VAST_PORT root@$VAST_HOST "systemctl restart casino-training"
