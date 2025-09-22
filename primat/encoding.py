# bloom_filter.py
import hashlib
from bitarray import bitarray
from typing import List, Iterable

class BloomFilterEncoder:
    """
    Encodes strings as Bloom filters with configurable parameters.
    """

    def __init__(
        self,
        size: int = 1024,
        num_hashes: int = 10,
        hash_seeds: List[int] = None,
        q: int = 2,
        pad: bool = True,
    ):
        """
        :param size: Length of the bit vector (m).
        :param num_hashes: Number of hash functions (k).
        :param hash_seeds: Optional distinct seeds for each hash.
        :param q: q-gram length.
        :param pad: Whether to pad input strings with q-1 '#' on each side.
        """
        self.size = size
        self.num_hashes = num_hashes
        self.q = q
        self.pad = pad

        if hash_seeds:
            assert len(hash_seeds) == num_hashes
            self.hash_seeds = hash_seeds
        else:
            # generate deterministic seeds
            self.hash_seeds = list(range(num_hashes))

    def _qgrams(self, s: str) -> Iterable[str]:
        """Generate q-grams from a string."""
        if self.pad:
            s = '#' * (self.q - 1) + s + '#' * (self.q - 1)
        return (s[i : i + self.q] for i in range(len(s) - self.q + 1))

    def _hashes(self, gram: str) -> List[int]:
        """Generate `num_hashes` indices from a q-gram."""
        return [
            int(hashlib.sha256(f"{gram}{seed}".encode()).hexdigest(), 16) % self.size
            for seed in self.hash_seeds
        ]

    def encode(self, s: str) -> bitarray:
        """
        Encode a string into its Bloom filter representation.
        :param s: Input string (e.g., concatenated fields).
        :returns: bitarray of length `size` with 1s at hashed q-gram positions.
        """
        bits = bitarray(self.size)
        bits.setall(False)
        for gram in self._qgrams(s):
            for idx in self._hashes(gram):
                bits[idx] = True
        return bits

    def encode_fields(self, fields: List[str]) -> List[bitarray]:
        """
        Encode multiple fields as separate Bloom filters.
        :param fields: List of strings (e.g., first name, surname, dob).
        :returns: List of bitarrays, one per field.
        """
        return [self.encode(f) for f in fields]

    def bloom_to_secondary(self, bf: bitarray, splits: int) -> List[tuple[int, int]]:
        """
        Generate "secondary encoding" as per recent research:
        – Partition Bloom filter into `splits` segments.
        – For each segment, return (sum of 1-bit positions, count of 1 bits).
        """
        if splits < 1 or splits > self.size:
            raise ValueError("Invalid number of splits")
        seg_len = self.size // splits
        sec = []
        for i in range(splits):
            start = i * seg_len
            end = start + seg_len if i < splits - 1 else self.size
            segment = bf[start:end]
            count = segment.count()
            # sum of positions relative to full bf:
            total = sum(start + idx for idx, val in enumerate(segment) if val)
            sec.append((total, count))
        return sec

if __name__ == "__main__":
    encoder = BloomFilterEncoder(size=512, num_hashes=5, q=2)
    bf = encoder.encode("Alice#Smith#1985-06-21")
    sec = encoder.bloom_to_secondary(bf, splits=4)
    print(f"Bloom filter (512 bits) has {bf.count()} 1s")
    print("Secondary encoding segments:", sec)