FROM python:buster

WORKDIR /code

# analyst is mostly I/O bound so we use the eventlet pool
RUN pip install celery[gevent,librabbitmq,redis]
RUN pip install numpy
RUN pip install dnspython

# CMD celery -A tasks worker -l debug --pool=gevent  --concurrency=16 --without-mingle
CMD celery -A tasks worker -l debug --concurrency=8 --prefetch=8 --without-mingle