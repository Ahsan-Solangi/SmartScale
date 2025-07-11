#!/bin/bash

echo "Running all unit tests..."

# Set PYTHONPATH to include the src directory
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

# Run data preprocessing tests
python3 -m unittest tests/test_data_preprocessing.py

# Run model training tests
python3 -m unittest tests/test_model_training.py

# Note: Model serving tests require the serving API to be running.
# For a full CI/CD, you'd start the serving container here, run tests, then stop it.
# For local development, ensure the serving container is up before running this script for serving tests.
# python3 -m unittest tests/test_model_serving.py

echo "All unit tests completed."


