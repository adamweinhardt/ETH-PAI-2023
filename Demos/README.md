# PAI Demos Repository 2023

### Setup: 
First create and activate an environment in conda. Run the following commands:

```bash
$ conda create -n pai-demos python=3.9 jupyter
$ conda activate pai-demos
```

Then, install the requirements. On the home directory run 
```bash
$ pip install -r requirements.txt
$ pip install -e git+https://github.com/jonasrothfuss/rllib.git#egg=rllib --no-deps
```

Start a jupyter notebook server.
```bash
$ jupyter notebook 
```

### Dockerfile (Thanks to Manuel Hässig)
You can also create a Docker container using:
```bash
$ docker build -t pai-demos . 
```

To run the docker container you can use:
```bash
$ docker run --network="bridge" -it --rm -p 8888:8888 pai-demos
```
Then open the link displayed on your terminal in your browser


For demos from previous years, please see the release [here](https://gitlab.inf.ethz.ch/OU-KRAUSE/pai-demos/-/releases/2020_2022_archive).
