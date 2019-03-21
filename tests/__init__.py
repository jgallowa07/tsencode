"""
Common code for the pyslim test cases.
"""
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division

import tsencode
import msprime
import pyslim
import random
import unittest
import base64
import os


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

    def assertTreeSequenceEqual(self,ts1,ts2):
        '''
        Here, we assert that the topology of two 
        tree sequences are equal

        msprime TreeSequence -> boolean
        '''

        for (edge_b,edge_d) in zip(ts1.edges(),ts2.edges()):
            self.assertEqual(edge_b,edge_d)

        for (node_b,node_d) in zip(ts1.nodes(),ts2.nodes()):
            self.assertEqual(node_b,node_d)

    def get_msprime_examples(self):
        """
        Generate some msprime simulations 
        across a range of parameters for encoding tests 
        """
        for n in [2, 10, 20]:
            for mutrate in [0.0]:
                for recrate in [1e-3,1e-4]:
                    for l in [1e2,1e3]:
                        yield msprime.simulate(n, length=l,mutation_rate=mutrate,
                                               recombination_rate=recrate)
