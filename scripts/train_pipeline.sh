#!/bin/bash

echo "Starting ML model training..."

# Build and run the training Docker container
docker-compose -f docker/docker-compose.yml build ml_training
docker-compose -f docker/docker-compose.yml run --rm ml_training

echo "ML model training completed."


