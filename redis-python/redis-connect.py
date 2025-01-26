import redis

# Create a Redis connection
r = redis.Redis(host='localhost', port=6379, decode_responses=True)

# Test the connection
print(r.ping())


# Set a key
r.set('key', 'value')

# Get the value
value = r.get('key')
print(value)  # Output: 'value'

all_keys = r.smembers(f"bike_vector_index_keys")
for key in all_keys:
    data = r.hgetall(key)
    print(data)