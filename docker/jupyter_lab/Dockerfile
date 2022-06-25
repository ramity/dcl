FROM python:buster

RUN pip install jupyterlab
RUN pip install numpy
RUN pip install matplotlib

WORKDIR /notebooks

ENTRYPOINT jupyter-lab . --port=8888 --ip=0.0.0.0 --allow-root