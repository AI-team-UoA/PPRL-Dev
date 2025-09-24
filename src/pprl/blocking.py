import random
import numpy as np
from bitarray import bitarray
from collections import defaultdict
from typing import List

class LSHBlocker:
    """
    LSH-based blocking for Bloom-filter-encoded records.
    Each record produces Λ blocking keys of length Ψ bits.
    """
    def __init__(
        self,
        psi: int = 36,
        lmbda: int = 3,
        prune_ratio: float = 0.5,
        prune_sample: int = 1000,
        seed: int = 42
    ):
        """
        :param psi: number of bit positions per key (Ψ).
        :param lmbda: number of keys per record (Λ).
        :param prune_ratio: ratio of most frequent/uncommon bit positions to prune.
        :param prune_sample: number of records to sample for frequency estimation.
        """
        self.psi = psi
        self.lmbda = lmbda
        self.prune_ratio = prune_ratio
        self.prune_sample = prune_sample
        self.seed = seed
        self.bit_positions = None  # will hold usable bit indices

    def _select_bit_positions(self, bflist: List[bitarray]) -> List[int]:
        """
        Determine a candidate pool of bit positions excluding too frequent or rare bits.
        Frequency estimated from initial sample.
        """
        m = len(bflist[0])
        freq = np.zeros(m, dtype=int)
        print(freq)
        sample = bflist if len(bflist) <= self.prune_sample else bflist[: self.prune_sample]
        for bf in sample:
            freq += np.array(bf.tolist(), dtype=int)
        print(freq)
        
        # normalize to frequency ratio
        freq_ratio = freq / len(sample)
        mask = (freq_ratio > self.prune_ratio) | (freq_ratio < (1 - self.prune_ratio))
        print(mask)
        # candidate positions are those not masked

        candidates = [i for i, bad in enumerate(mask) if not bad]
        print(candidates)

        if len(candidates) < self.psi:
            raise ValueError("Not enough bit positions after pruning")
        return candidates

    def fit(self, bflist: List[bitarray]):
        """Initialize blocker by sampling to prune and selecting usable bit positions."""
        random.seed(self.seed)
        self.bit_positions = self._select_bit_positions(bflist)

    def block_record(self, bf: bitarray) -> List[str]:
        """
        Generate Λ blocking keys for a single Bloom filter record.
        Each key is Ψ bits sampled from bf at selected positions.
        """
        if self.bit_positions is None:
            raise RuntimeError("Call fit() before blocking")
        keys = []
        for i in range(self.lmbda):
            positions = random.sample(self.bit_positions, self.psi)
            bits = ''.join('1' if bf[j] else '0' for j in positions)
            # optionally, can hash this bit string to reduce size
            keys.append(bits)
        return keys

    def block_all(self, bflist: List[bitarray]) -> dict:
        """
        Generate blocking map: key -> list of record indices.
        """
        blocks = defaultdict(list)
        for idx, bf in enumerate(bflist):
            for key in self.block_record(bf):
                blocks[key].append(idx)
        return blocks