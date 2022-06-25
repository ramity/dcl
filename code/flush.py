import redis

redis = redis.Redis(
    host = 'scheduler_redis',
    port = 6379,
    db = 0
)
redis.flushall()