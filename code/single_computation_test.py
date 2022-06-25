import redis
import numpy
import tasks
import math
import uuid


# Config values
redis_max_size = 5.12 * (10 ** 8)
a_shape = (60000,)
b_shape = (256,)
c_shape = (60000, 256)


# Connect to scheduler redis instance
redis = redis.Redis(
    host = 'scheduler_redis',
    port = 6379,
    db = 0
)


# Generate computation (float64s)
a = numpy.random.rand(*a_shape)
a_bytes = a.tobytes()
a_byte_len = len(a_bytes)
a_chunks = math.ceil(a_byte_len / redis_max_size)
a_keys = []
for chunk_index in range(a_chunks):
    chunk_key = str(uuid.uuid4())
    chunk_start_index = int(chunk_index * redis_max_size)
    chunk_stop_index = int((chunk_index + 1) * redis_max_size)
    if chunk_stop_index > a_byte_len:
        chunk_stop_index = a_byte_len
    redis.set(chunk_key, a_bytes[chunk_start_index:chunk_stop_index])
    a_keys.append(chunk_key)

b = numpy.random.rand(*b_shape)
b_bytes = b.tobytes()
b_byte_len = len(b_bytes)
b_chunks = math.ceil(b_byte_len / redis_max_size)
b_keys = []
for chunk_index in range(b_chunks):
    chunk_key = str(uuid.uuid4())
    chunk_start_index = int(chunk_index * redis_max_size)
    chunk_stop_index = int((chunk_index + 1) * redis_max_size)
    if chunk_stop_index > b_byte_len:
        chunk_stop_index = b_byte_len
    redis.set(chunk_key, b_bytes[chunk_start_index:chunk_stop_index])
    b_keys.append(chunk_key)


# Create task that references the keys for the byte memory stored in redis
context = tasks.outer_computation.delay(a_keys, b_keys, a_shape, b_shape)
c_keys = context.get()
print(c_keys)
context.forget()


c_bytes = bytearray()
for c_key in c_keys:
    c_bytes.extend(redis.get(c_key))
c = numpy.frombuffer(c_bytes, dtype=numpy.float64).reshape(*c_shape)


# verify locally computed value and remote computed value are the same
local_c = numpy.outer(a, b)
c_check = numpy.array_equiv(c, local_c)
print(c_check)

# delete data
redis.flushall()
# redis.delete(*a_keys)
# redis.delete(*b_keys)
# redis.delete(*c_keys)