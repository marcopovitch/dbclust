# Build image
docker build -t dbclust:latest .

# Run 
docker run -it --rm -v ${HOME}/github/dbclust:/tmp/dbclust -w /tmp/dbclust dbclust bash
