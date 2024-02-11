# Build image
docker build -t dbclust:latest .

# Run

User have to provide:
- nll time grid files
- nll template files / configuration files


```
docker run -it --rm \
    -v ${HOME}/github/dbclust:/tmp/dbclust \
    -v  ${HOME}/Dockers/routine/nll/data/times:/nll_times \
    -w /tmp/dbclust \
    dbclust \
    bash
