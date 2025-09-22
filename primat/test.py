from bloom_filter import BloomFilterEncoder
from blocking import LSHBlocker

# toy example
records = [
    {"fname": "Alice", "lname": "Smith", "dob": "1985-06-21"},
    {"fname": "Alyce", "lname": "Smyth", "dob": "1985-06-21"},
]
    
# Instantiate encoder and blocker
encoder = BloomFilterEncoder(size=512, num_hashes=5, q=2)
blocker = LSHBlocker(psi=36, lmbda=3)

# Prepare data
def join_fields(rec, fields): return '#'.join(rec[f] for f in fields)

fields = ["fname", "lname", "dob"]
bloom_filters = [encoder.encode(join_fields(r, fields)) for r in records]

# Fit blocker on bloom filters (for pruning + key generation)
blocker.fit(bloom_filters)

# Generate blocking map
blocking_map = blocker.block_all(bloom_filters)

# Show example blocks
for key, ids in blocking_map.items():
    print(f"Key {key}: records {ids}")