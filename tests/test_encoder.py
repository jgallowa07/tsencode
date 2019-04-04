
from __future__ import print_function
from __future__ import division

import tsencode
import tests
import numpy as np


np.random.seed(23)


class TestEncoding(tests.TsEncodeTestCase):
    """
    Tests for the TsEncoder class.
    """

    def test_initialize_layer_msprime(self):
        """
        Let's test that the correct dimentions are created for each new layer
        """

        for ts in self.get_msprime_examples():
            num_nodes = ts.num_nodes
            sequence_length = ts.sequence_length
            encoder = tsencode.TsEncoder(treeSequence=ts)
            encoder.initialize_layer()
            encoding = encoder.get_encoding()

            self.assertEqual(encoding.shape[0], num_nodes)
            self.assertEqual(encoding.shape[1], sequence_length)
            self.assertEqual(encoding.shape[2], 1)

    def test_initialize_layer_slim(self):
        """
        Let's test that the correct dimentions are created for each new layer
        """
        for ts in self.get_slim_examples():
            num_nodes = ts.num_nodes
            sequence_length = ts.sequence_length
            encoder = tsencode.TsEncoder(treeSequence=ts)
            encoder.initialize_layer()
            encoding = encoder.get_encoding()

            self.assertEqual(encoding.shape[0], num_nodes)
            self.assertEqual(encoding.shape[1], sequence_length)
            self.assertEqual(encoding.shape[2], 1)

    def test_general_prop_layer_dim(self):
        """
        check that the shape is correct for a trivial tree sequence
        with three dimentional prop layer representing spatial location
        """
        ts = self.get_trivial_ts()
        encoder = tsencode.TsEncoder(ts)
        weights = [[1, 1, 1], [2, 2, 2], [1, 1, 2]]
        encoder.add_prop_layer(weights, np.sum)
        encoding = encoder.get_encoding()
        self.assertEqual(encoding.shape[0], 5)
        self.assertEqual(encoding.shape[1], 10)
        self.assertEqual(encoding.shape[2], 3)

    def test_prop_with_sum_msprime(self):
        """
        Run through all msprime examples and test that the
        value of the root node row, for all sparse trees,
        is equal to the the sum of the initialized weights
        for all samples.
        """

        for ts in self.get_msprime_examples():
            encoder = tsencode.TsEncoder(ts)
            random_init_weights = np.random.random_integers(0, 10, [3, ts.num_samples])
            encoder.add_prop_layer(random_init_weights, np.sum)
            encoding = encoder.get_encoding()
            for st in ts.trees():
                root = st.root
                left_interval = st.interval[0]
                right_interval = st.interval[1]
                first_column_index_of_root = encoder.map_locus_to_column(left_interval)
                last_column_index_of_root = encoder.map_locus_to_column(right_interval)
                if first_column_index_of_root == last_column_index_of_root:
                    continue
                for dim in range(3):
                    fci = first_column_index_of_root
                    encoding_value_for_root = encoding[root, fci, dim]
                    evr = encoding_value_for_root
                    self.assertEqual(evr, sum(random_init_weights[dim]))

    def test_prop_layer_trivial_ts(self):
        """
        This is the sanity check that the propagation algorithm in
        tsencode/helpers.py is producing the right results in a very simple encoding.

        The results found tests/trivial_tree_tables/trivial_tree_encoding.npz
        were checked by hand to ensure correctness, and here we compare those results
        to those being produced by the TsEncoder class.
        """
        ts = self.get_trivial_ts()
        encoder = tsencode.TsEncoder(ts)
        weights = [[1, 1, 1], [9, 5, 2], [1, 1, 2]]
        encoder.add_prop_layer(weights, np.sum)
        encoding = encoder.get_encoding()

        correct_encoding = np.load("tests/trivial_tree_tables/trivial_encoding.npz")
        self.assertTrue(np.array_equal(encoding, correct_encoding))
