from urllib import request
import gzip
import numpy
import pickle
import celery
import redis
import tasks
import sys
import os
import uuid
import math


redis_max_size = 5.12 * (10 ** 8)


# MNIST configuration data 
filename = [
    ['training_images', 'train-images-idx3-ubyte.gz'],
    ['test_images', 't10k-images-idx3-ubyte.gz'],
    ['training_labels', 'train-labels-idx1-ubyte.gz'],
    ['test_labels', 't10k-labels-idx1-ubyte.gz']
]
minst_directory = './mnist/'


# MNIST helper functions
def download_mnist():
    print('[ ] Download starting.')
    base_url = 'http://yann.lecun.com/exdb/mnist/'
    for name in filename:
        print('[-] Downloading {}...'.format(name[0]))
        request.urlretrieve(base_url + name[1], minst_directory + name[1])
    print('[x] Download complete.')


def save_mnist():
    print('[ ] Saving downloaded files as pickled object.')
    mnist = {}
    for name in filename[:2]:
        with gzip.open(minst_directory + name[1], 'rb') as f:
            mnist[name[0]] = numpy.frombuffer(f.read(), numpy.uint8, offset = 16).reshape(-1, 28 * 28)
    for name in filename[-2:]:
        with gzip.open(minst_directory + name[1], 'rb') as f:
            mnist[name[0]] = numpy.frombuffer(f.read(), numpy.uint8, offset = 8)
    with open(minst_directory + 'mnist.pkl', 'wb') as f:
        pickle.dump(mnist, f)
    print('[x] Save complete.')


def init():
    if os.path.exists(minst_directory + 'mnist.pkl'):
        print('[@] MNIST data detected; skipping download.')
        return
    download_mnist()
    save_mnist()


def load():
    print('[ ] Loading MNIST data.')
    with open(minst_directory + 'mnist.pkl','rb') as f:
        mnist = pickle.load(f)
    print('[x] Loaded.')
    return mnist['training_images'], mnist['training_labels'], mnist['test_images'], mnist['test_labels']


# Initialize and load the mnist dataset
init()
train_X, train_labels, test_X, test_labels = load()


# Convert inputs to float64 + perform normalization
train_X = train_X.astype('float64') / 255
test_X = test_X.astype('float64') / 255
train_X_mean = train_X.mean()
test_X_mean = test_X.mean()
train_X_std = train_X.std()
test_X_std = test_X.std()
train_X -= train_X_mean
test_X -= test_X_mean
train_X /= train_X_std
test_X /= test_X_std


# Convert values to one-hot-encoded vectors
train_Y = numpy.zeros((train_labels.size, train_labels.max() + 1))
train_Y[numpy.arange(train_labels.size), train_labels] = 1
test_Y = numpy.zeros((test_labels.size, test_labels.max() + 1))
test_Y[numpy.arange(test_labels.size), test_labels] = 1


# Model hyper parameters
learning_rate = 0.05
seed = 402
epochs = 1000
numpy.random.seed(seed)


# Describing input sizes
training_images_count = 60000
testing_images_count = 10000
training_features_count = 784
testing_features_count = 784


# Layer 1 model values
w1 = numpy.random.randn(784, 128) * 0.01
b1 = numpy.zeros(128)
z1 = numpy.zeros(128)
a1 = numpy.zeros(128)

print('[@] layer 1 requires: {:.2f} GB memory'.format((training_images_count * 128 * w1.itemsize * testing_features_count) / (10 ** 9)))


# Layer 2 model values
w2 = numpy.random.randn(128, 10) * 0.01
b2 = numpy.zeros(10)
z2 = numpy.zeros(10)
a2 = numpy.zeros(10)


# Connect to scheduler redis instance
print('[ ] Connecting to redis.')
redis = redis.Redis(
    host = 'scheduler_redis',
    port = 6379,
    db = 0
)
# Make sure we start with a fresh redis
redis.flushall()
print('[x] Connected.')


# Compiling layer 1 computations
layer_1_computations = []
print('[ ] Partitioning and distributing layer 1 computations.')

for index in range(training_features_count):

    print('[-] Progress: {}/{} ({:.2f}%)'.format(index + 1, training_features_count, ((index + 1) / training_features_count) * 100), end='\r')

    # Partitions and data describing it
    input_partition = train_X[:, index]
    weight_partition = w1[index]
    input_enshuffle = numpy.arange(training_images_count)
    input_unshuffle = numpy.arange(training_images_count)
    weight_enshuffle = numpy.arange(128)
    weight_unshuffle = numpy.arange(128)

    # Shuffle enshuffle arrays and calculate unshuffle indicies
    numpy.random.shuffle(input_enshuffle)
    numpy.random.shuffle(weight_enshuffle)
    for index in input_enshuffle:
        input_unshuffle[input_enshuffle[index]] = index
    for index in weight_enshuffle:
        weight_unshuffle[weight_enshuffle[index]] = index

    # Chunk input and set on redis
    input_bytes = input_partition[input_enshuffle].tobytes()
    input_keys = []
    input_byte_len = len(input_bytes)
    input_chunks = math.ceil(input_byte_len / redis_max_size)
    for chunk_index in range(input_chunks):
        chunk_key = str(uuid.uuid4())
        chunk_start_index = int(chunk_index * redis_max_size)
        chunk_stop_index = int((chunk_index + 1) * redis_max_size)
        if chunk_stop_index > input_byte_len:
            chunk_stop_index = input_byte_len
        redis.set(chunk_key, input_bytes[chunk_start_index:chunk_stop_index])
        input_keys.append(chunk_key)

    # Chunk weight and set on redis
    weight_bytes = weight_partition[weight_enshuffle].tobytes()
    weight_keys = []
    weight_byte_len = len(weight_bytes)
    weight_chunks = math.ceil(weight_byte_len / redis_max_size)
    for chunk_index in range(weight_chunks):
        chunk_key = str(uuid.uuid4())
        chunk_start_index = int(chunk_index * redis_max_size)
        chunk_stop_index = int((chunk_index + 1) * redis_max_size)
        if chunk_stop_index > weight_byte_len:
            chunk_stop_index = weight_byte_len
        redis.set(chunk_key, weight_bytes[chunk_start_index:chunk_stop_index])
        weight_keys.append(chunk_key)

    # Create computation object and add to layer 1 computations
    computation = {}
    computation['input_keys'] = input_keys
    computation['weight_keys'] = weight_keys
    computation['input_shape'] = input_partition.shape
    computation['weight_shape'] = weight_partition.shape
    computation['input_len'] = input_byte_len
    computation['weight_len'] = weight_byte_len
    computation['input_enshuffle'] = input_enshuffle
    computation['input_unshuffle'] = input_unshuffle
    computation['weight_enshuffle'] = weight_enshuffle
    computation['weight_unshuffle'] = weight_unshuffle
    layer_1_computations.append(computation)

    # Free up some memory
    del input_partition
    del weight_partition
    del input_enshuffle
    del input_unshuffle
    del weight_enshuffle
    del weight_unshuffle


# Create celery invocations of dot operation
print('\n[x] Completed.')
layer_1_signatures = []
print('[ ] Creating celery signatures.')

for index in range(training_features_count):

    print('[-] Progress: {}/{} ({:.2f}%)'.format(index + 1, training_features_count, ((index + 1) / training_features_count) * 100), end='\r')

    a = {
        'keys': layer_1_computations[index]['input_keys'],
        'shape': layer_1_computations[index]['input_shape'],
        'len': layer_1_computations[index]['input_len']
    }

    b = {
        'keys': layer_1_computations[index]['weight_keys'],
        'shape': layer_1_computations[index]['weight_shape'],
        'len': layer_1_computations[index]['weight_len']
    }

    context = tasks.outer_computation.signature(args=(a, b))
    layer_1_signatures.append(context)


# Make sure all keys exist
# keys_missing = False
# for index in range(training_features_count):
#     computation = layer_1_computations[index]
#     if not redis.exists(computation['input_key']) or not redis.exists(computation['weight_key']):
#         keys_missing = True
#         break
#
# if keys_missing:
#     print('Keys missing. Flushing all dbs.')
#     redis.flushall()
#     sys.exit(1)


# Schedule celery tasks
print('\n[x] Completed.')
print('[ ] Pushing computations to celery and awaiting results.')

layer_1_group = celery.group(layer_1_signatures)
layer_1_promise = layer_1_group.apply_async()
layer_1_results = layer_1_promise.join()
layer_1_promise.forget()


# Free some memory
del layer_1_signatures
del layer_1_group
del layer_1_promise


# TODO: TEST CPUBOUND PREFORK SETUP BY REBUILDING + EXPORT THE UNSHUFFLING PROCESS TO A TASK



# Pull results from redis
print('\n[x] Completed.')
print('[ ] Pull computation results from redis')
c_shape = (layer_1_computations[0]['input_shape'][0], layer_1_computations[0]['weight_shape'][0])
c_sum = numpy.zeros(c_shape, dtype=numpy.float64)

for index in range(training_features_count):

    print('[-] Progress: {}/{} ({:.2f}%)'.format(index + 1, training_features_count, ((index + 1) / training_features_count) * 100), end='\r')

    layer_1_result = layer_1_results[index]
    computation = layer_1_computations[index]
    input_unshuffle = computation['input_unshuffle']
    weight_unshuffle = computation['weight_unshuffle']

    # Merge chunks
    c_bytes = bytearray()
    for key in layer_1_result:
        c_bytes.extend(redis.get(key))
    c = numpy.frombuffer(c_bytes, dtype=numpy.float64).reshape(*c_shape)
    
    # Add unshuffled weighted sum matrix to c_sum
    c_sum += c[input_unshuffle, :][:, weight_unshuffle]


local_c = numpy.dot(train_X, w1)
c_check = numpy.array_equiv(c_sum, local_c)
print(c_sum)
print(local_c)
print(c_check)


# Clean up redis
print('\n[x] Completed.')
print('[ ] Flushing all dbs.')
redis.flushall()
print('[x] Completed.')