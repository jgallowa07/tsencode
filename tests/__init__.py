"""
Common code for the pyslim test cases.
"""
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division

import msprime
import pyslim
import tskit
import unittest


_slim_example_files = [
    "tests/slim_trees/high_dispersal_2d_slim.trees",
    "tests/slim_trees/neutral_slim.trees",
    "tests/slim_trees/sweep_slim.trees"]

_trivial_tables = [
    "tests/trivial_tree_tables/nodes.txt",
    "tests/trivial_tree_tables/edges.txt"]


def setUp():
    pass


def tearDown():
    pass


class TsEncodeTestCase(unittest.TestCase):
    '''
    Base class for test cases in tsencode.
    '''

    def assertArrayEqual(self, x, y):
        self.assertListEqual(list(x), list(y))

    def assertArrayAlmostEqual(self, x, y):
        self.assertEqual(len(x), len(y))
        for a, b in zip(x, y):
            self.assertAlmostEqual(a, b)

    def assertTreeSequenceEqual(self, ts1, ts2):
        '''
        Here, we assert that the topology of two
        tree sequences are equal

        msprime TreeSequence -> boolean
        '''

        for (edge_b, edge_d) in zip(ts1.edges(), ts2.edges()):
            self.assertEqual(edge_b, edge_d)

        for (node_b, node_d) in zip(ts1.nodes(), ts2.nodes()):
            self.assertEqual(node_b, node_d)

    def get_msprime_examples(self):
        """
        Generate some msprime simulations
        across a range of parameters for encoding tests
        """
        seed = 23
        for n in [10, 20]:
            for recrate in [0, 1e-3]:
                for l in [1e2, 1e3]:
                    yield msprime.simulate(
                        n, length=l, mutation_rate=0,
                        recombination_rate=recrate, random_seed=seed)

    def get_slim_examples(self):
        """
        Load some slim tree sequences
        for testing.
        """
        for filename in _slim_example_files:
            yield pyslim.load(filename)

    def get_trivial_ts(self):
        """
        load in a trivial tree seuqence for testing purposes
        this is the same topology of a tree sequence example found in
        https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006581
        """
        nodes = open(_trivial_tables[0], "r")
        edges = open(_trivial_tables[1], "r")
        ts = tskit.load_text(nodes=nodes, edges=edges, strict=False)
        return ts
