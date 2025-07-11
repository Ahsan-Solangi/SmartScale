
#!/bin/bash

echo "Starting ML model deployment..."

# Build and run the serving Docker container
docker-compose -f docker/docker-compose.yml build ml_serving
docker-compose -f docker/docker-compose.yml up -d ml_serving

echo "ML model deployment completed. The model serving API should be available on port 5000."


