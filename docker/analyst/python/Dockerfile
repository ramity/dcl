FROM python:buster

WORKDIR /code

# analyst is mostly I/O bound so we use the eventlet pool
RUN pip install celery[gevent,librabbitmq,redis]
RUN pip install numpy
RUN pip install dnspython

ENTRYPOINT tail -f /dev/null