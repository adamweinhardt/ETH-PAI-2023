# set base image
FROM jupyter/minimal-notebook:python-3.7.12 

# requirements
COPY requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt
RUN pip install -e git+https://github.com/jonasrothfuss/rllib.git#egg=rllib --no-deps

# set permissions
RUN fix-permissions $CONDA_DIR
RUN fix-permissions /home/$NB_USER

# add directories to image
ADD demos demos 
ADD data data
ADD style style

# update dependencies
USER root
RUN apt-get update && apt-get install -y libgl1 
RUN apt-get install -y build-essential libopenblas-dev python3-opengl xvfb xauth

# update permissions
RUN chmod -R 777 demos
USER $NB_UID
