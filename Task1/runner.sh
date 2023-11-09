docker build --tag task1 .
docker run -v "$( cd "$( dirname "$0" )" && pwd )":/results task1
