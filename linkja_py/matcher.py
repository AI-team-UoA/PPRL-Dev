from pybloom_live import BloomFilter
from rbloom import Bloom

# def dice_coefficient(bf1: BloomFilter, bf2:BloomFilter):
#     bit1 = bf1.bitarray
#     bit2 = bf2.bitarray
    
#     intersection = (bit1 & bit2).count()
#     bf1_len = bit1.count()
#     bf2_len = bit2.count()
#     return 2 * intersection / (bf1_len + bf2_len)
    
    
def dice_coefficient(bf1: Bloom, bf2:Bloom):
    intersection = (bf1 & bf2)
    return 2 * intersection.approx_items / (bf1.approx_items + bf2.approx_items)
