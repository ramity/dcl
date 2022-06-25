from celery.utils.log import get_task_logger
logger = get_task_logger(__name__)
from celery import Celery
from redis import Redis
import time
import numpy
import math
import uuid

app = Celery('DCL', backend='redis://scheduler_redis/0', broker='amqp://guest:guest@scheduler_rabbitmq//')
redis_max_size = 5.12 * (10 ** 8)
task_routes = {
    'DCL.outer_computation': 'untrusted',
    'DCL.unshuffle_outer_computation_result': 'trusted'
}


@app.task
def test():
    return "hello world"


@app.task(bind=True)
def outer_computation(self, a_obj, b_obj):

    # obj:
    # - keys
    # - shape
    # - len

    a_keys = 
    b_keys = b_obj['b_keys']
    a_shape = a_obj['a_shape'], b_shape

    connection_start = time.perf_counter_ns()
    redis = Redis(
        host = 'scheduler_redis',
        port = 6379,
        db = 0
    )
    connection_end = time.perf_counter_ns()
    connection_delta = connection_end - connection_start

    redis_get_start = time.perf_counter_ns()
    a_bytes = bytearray(a_obj['len'])
    for a_key in a_obj['keys']:
        a_bytes.extend(redis.get(a_key))
    b_bytes = bytearray(b_obj['len'])
    for b_key in b_obj['keys']:
        b_bytes.extend(redis.get(b_key))
    redis_get_end = time.perf_counter_ns()
    redis_get_delta = redis_get_end - redis_get_start

    decode_start = time.perf_counter_ns()
    a = numpy.frombuffer(a_bytes, dtype=numpy.float64).reshape(*a_shape)
    b = numpy.frombuffer(b_bytes, dtype=numpy.float64).reshape(*b_shape)
    decode_end = time.perf_counter_ns()
    decode_delta = decode_end - decode_start

    compute_start = time.perf_counter_ns()
    c = numpy.outer(a, b)
    compute_end = time.perf_counter_ns()
    compute_delta = compute_end - compute_start
    
    delete_start = time.perf_counter_ns()
    redis.delete(*a_obj['keys'])
    redis.delete(*b_obj['keys'])
    delete_end = time.perf_counter_ns()
    delete_delta = delete_end - delete_start

    encode_start = time.perf_counter_ns()
    c_bytes = c.tobytes()
    encode_end = time.perf_counter_ns()
    encode_delta = encode_end - encode_start
    
    redis_set_start = time.perf_counter_ns()
    c_bytes_len = len(c_bytes)
    c_chunks = math.ceil(c_bytes_len / redis_max_size)
    c_keys = []
    for chunk_index in range(c_chunks):
        chunk_key = str(uuid.uuid4())
        chunk_start_index = int(chunk_index * redis_max_size)
        chunk_stop_index = int((chunk_index + 1) * redis_max_size)
        if chunk_stop_index > c_bytes_len:
            chunk_stop_index = c_bytes_len
        redis.set(chunk_key, c_bytes[chunk_start_index:chunk_stop_index])
        c_keys.append(chunk_key)
    redis_set_end = time.perf_counter_ns()
    redis_set_delta = redis_set_end - redis_set_start

    logger.debug('connection:{}, redis_get:{}, decode:{}, compute:{}, delete:{}, encode:{}, redis_set:{}'.format(
        connection_delta,
        redis_get_delta,
        decode_delta,
        compute_delta,
        delete_delta,
        encode_delta,
        redis_set_delta
    ))

    return c_keys


@app.task(bind=True)
def unshuffle_outer_computation_result(self, c_keys, c_shape, input_unshuffle_keys, weight_unshuffle_keys):

    c_bytes = bytearray()
    