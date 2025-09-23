# from pybloom_live import BloomFilter
from rbloom import Bloom as BloomFilter
from pseudonymizer import pseudonymize

def encode_string_to_bloom(s: str, bloom_size=1000, k=5):
    s = s.lower().strip()
    bf = BloomFilter(bloom_size, 0.001)
    for i in range(len(s) - k + 1):
        bf.add(s[i:i+k])
    return bf
