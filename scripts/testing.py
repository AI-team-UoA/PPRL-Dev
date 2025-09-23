from rbloom import Bloom
from pybloom_live import BloomFilter
import timeit
import random
import string
import sys

# Helper to generate random strings
def random_string(length=10):
    return ''.join(random.choice(string.ascii_letters) for _ in range(length))


if __name__ == '__main__': 
   
   