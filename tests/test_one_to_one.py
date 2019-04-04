"""
Test cases for the metadata reading/writing of pyslim.
"""
from __future__ import print_function
from __future__ import division

import tsencode
from tsencode.helpers import GlueInt8
import tskit
import tests
import numpy as np


class TestOneToOneMapping(tests.TsEncodeTestCase):
    """
    Tests for the one-to-one encoding scheme
    """

    def test_Inverse(self):

        for ts in self.get_msprime_examples():
            dts = self.DiscretizeTreeSequence(ts)
            # edts1 = tsencode.encode(dts, return_8bit=False)
            encoder = tsencode.TsEncoder(dts)
            encoder.add_one_to_one()
            edts = encoder.get_encoding()
            de_dts = self.DecodeTree(edts)
            self.assertTreeSequenceEqual(dts, de_dts)

    def DecodeTree(self, A):
        """
        Take in the array produced by 'EncodeTreeSequence()' and return a
        the inverse operation to produce a TreeSequence() for testing.
        """

        num_rows = A.shape[0]
        num_columns = A.shape[1]
        tables = tskit.TableCollection(sequence_length=num_columns)
        node_table = tables.nodes
        edge_table = tables.edges
        pop_table = tables.populations
        pop_table.add_row()
        for row in range(num_rows):
            flag = 0
            time = A[row, 0, 0]
            if(time == 0.0):
                flag = 1
            node_table.add_row(flags=flag, time=float(time), population=0)
            for column in range(num_columns):
                top = A[row, column, 1]
                bot = A[row, column, 2]
                # for padding, we don't add edges
                if((top < 0) | (bot < 0)):
                    continue
                parent = GlueInt8(top, bot)
                edge_table.add_row(left=column, right=column + 1, parent=parent, child=row) # NOQA
        tables.sort()
        tables.simplify()
        ts = tables.tree_sequence()
        return ts

    def DiscretizeTreeSequence(self, ts):
        """
        Disretise float values within a tree sequence
        mainly for testing purposes to make sure the decoding is equal to pre-encoding.
        """

        tables = ts.dump_tables()
        nodes = tables.nodes
        edges = tables.edges
        oldest_time = max(nodes.time)
        nodes.set_columns(flags=nodes.flags,
                          time=(nodes.time / oldest_time) * 256,
                          population=nodes.population)
        edges.set_columns(left=np.round(edges.left),
                          right=np.round(edges.right),
                          child=edges.child,
                          parent=edges.parent)
        return tables.tree_sequence()
