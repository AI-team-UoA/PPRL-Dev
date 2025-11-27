from unittest.mock import MagicMock

from bitarray import bitarray
import pytest

# Assuming your classes are accessible, e.g., from your_module
from pprl.blocking import AbstractBlockBuilding, LSHBlocker, BitBlocker, FAISSBlocking

# If they are in the same file, you can import them directly.


def create_mock_encoded_data(attributes= None, length=12):
    """ --- Helper function to create the mock data structure --- """

    mock_data = MagicMock()
    attributes = ['a', 'b'] if not attributes else attributes
    mock_data.attributes = attributes
    mock_data.length = length
    # Simulate two datasets: records 0-2 (DS1) and 3-4 (DS2)
    mock_data.bounds = [0, 3, 6]

    # Create dummy bitarray_dict
    bitarray_dict = {}
    bfs = ['0000', '0001', '0010']

    for i in range(3):
        # Create a simple, deterministic bitarray for testing
        # e.g., all 0s for index < 3, all 1s for index >= 3
        dummy_bf = bitarray(bfs[i] * int(length / 4))
        bitarray_dict[i] = {attr: dummy_bf for attr in attributes}
        bitarray_dict[i+3] = {attr: dummy_bf for attr in attributes}


    mock_data.bitarray_dict = bitarray_dict

    # Mock ground truth for evaluation tests
    mock_data.skip_ground_truth = False
    mock_data.ground_truth = MagicMock() # Mock the dataframe/iterator

    # For FAISSBlocking evaluate
    mock_data.ground_truth.iterrows.return_value = [
        (0, (0,0)),
        (1, (1,1)),
        (2, (2,2))
    ]

    mock_data.ground_truth.itertuples.return_value = [
        (0, 0),
        (1,1),
        (2,2)
    ]

    mock_data.ground_truth.__len__.return_value = 3 # Or any other iterable with 3 elements

    return mock_data


# class TestAbstract:
#     """AbstractBlockBuilding"""

    # def test_report(self):
    #     blokcer = AbstractBlockBuilding()
    #     blocker.report()


# --- Test class for LSHBlocker ---
class TestLSHBlocker:
    """Test lsh blocker"""


    def test_lsh_build_blocks_flow(self):
        """Test for lsh"""

        # ARRANGE
        blocker = LSHBlocker(psi=4, lmbda=2, seed=1, prune_ratio=0.9)
        mock_data = create_mock_encoded_data( attributes=['a', 'b'])
        # # Ensure _select_bit_positions is stable
        # mocker.patch.object(blocker, '_select_bit_positions',
        #                     return_value=list(range(5, 15)))

        # ACT
        with pytest.raises(AttributeError) as e_info: 
            blocker.report()

        blocks = blocker.build_blocks(mock_data, attributes=['a', 'b'])
        blocker.report()

        # ASSERT
        # 1. Check if the block structure is correct
        assert isinstance(blocks, dict)
        # 2. Check if the blocks contain data from both datasets
        # Since all DS1 records are '000...' and DS2 records are '111...',
        # and we set a fixed seed/pruned positions, LSH keys should be consistent.
        # Expect at least two blocks, one for DS1 and one for DS2's distinct keys.

        # The keys for DS1 should be consistent, and for DS2 should be consistent,
        # but the keys for DS1 should be different from DS2 keys.
        # Check that blocks are correctly "cleaned" (contain pairs)
        for block in blocks.values():
            assert len(block[0]) > 0 and len(block[1]) > 0

# --- Test class for BitBlocker ---
class TestBitBlocker:
    """TestBitBlocker"""

    def test_bitblocker_fit(self):
        """Bit Blocker fit"""
        # ARRANGE
        blocker = BitBlocker(psi=4, lmbda=2, seed=1)
        mock_data = create_mock_encoded_data(attributes=['a', 'b'], length=12)

        # ACT
        blocker.build_blocks(mock_data, attributes=['a', 'b'])

        # ASSERT
        # Check if hash_len is correct (10 bits * 2 attributes = 20)
        assert blocker.hash_len == 24
        # Check if hash_indices has the right dimensions (lmbda x psi)
        assert len(blocker.hash_indices) == 2
        assert len(blocker.hash_indices[0]) == 4
        # Check that indices are within the correct range [0, 20)
        assert all(0 <= idx < 24 for indices in blocker.hash_indices for idx in indices)

    def test_evaluate(self):
        """Evaluation testing"""
        mock_data = create_mock_encoded_data(attributes=['a', 'b'], length=12)
        predicted = {"000" : {0:{0}, 1:{3}},
                     "111" : {0:{1}, 1:{4}},
                     "2" : {0:{2}, 1:{5}},}
        blocker = BitBlocker(psi=4, lmbda=2, seed=1)
        _ = blocker.build_blocks(mock_data)
        ev = blocker.evaluate(predicted)


        assert ev['F1 %'] == 100.0


class TestFAISS:
    """"TEST FAISS"""

    def test_faiss(self):
        """"TEST FAISS"""
        #  ARRANGE
        blocker = FAISSBlocking()

        mock_data = create_mock_encoded_data(length=24, attributes=['a', 'b'])


        # ACT
        blocks = blocker.build_blocks(encoded_data=mock_data, attributes=['a', 'b'], top_k=1)

        assert blocks[0] == {3}
        mock_data = create_mock_encoded_data(length=24, attributes=['b'])
        blocks = blocker.build_blocks(encoded_data=mock_data, attributes=['b'], top_k=1)

        assert blocks[0] == {3}

        ev = blocker.evaluate(blocks)
        print(blocks)

        assert ev['F1 %'] == 100.0
        