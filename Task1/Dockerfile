FROM mambaorg/micromamba:latest
COPY env.yaml /tmp/env.yaml
USER root
RUN apt-get update && apt-get install -y gcc && apt-get install -y g++
RUN micromamba install -y -n base -f /tmp/env.yaml && \
    micromamba clean --all --yes
ARG MAMBA_DOCKERFILE_ACTIVATE=1  # (otherwise python will not be found)
ARG PYTHONUNBUFFERED=1
RUN pip install pyarmor==6.7.4
WORKDIR /code
ADD * /code/
ADD pytransform /code/pytransform
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/code/pytransform
WORKDIR /code
CMD python -u checker_client.py
