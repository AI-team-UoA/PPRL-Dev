from bloom_filter import BloomFilterEncoder
from blocking import LSHBlocker
from bitarray import bitarray
from typing import List

class LinkagePipeline:
    def __init__(self, encoder: BloomFilterEncoder, blocker: LSHBlocker, fields: List[str]):
        self.encoder = encoder
        self.blocker = blocker
        self.fields = fields

    def encode_records(self, records: List[dict]) -> List[bitarray]:
        return [self.encoder.encode('#'.join(r[f] for f in self.fields)) for r in records]

    def block(self, records: List[dict]) -> dict:
        bloom_filters = self.encode_records(records)
        self.blocker.fit(bloom_filters)
        return self.blocker.block_all(bloom_filters)