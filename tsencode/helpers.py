"""
This file contains helper functions for Encoding an msprime Tree Sequence as part
of tsEncode.
"""

import msprime
import numpy as np
import itertools


def get_genome_coordinates(ts, dim=1):
    """
    Loop through the tree sequence and
    get all coordinates of each genome of diploid individual.

    :param dim: the dimentionality, 1,2, or 3.
    """

    coordinates = []
    for i in range(dim):
        coordinates.append(np.zeros(ts.num_samples))
    for ind in ts.individuals():
        for d in range(dim):
            for geno in range(2):
                coordinates[d][ind.nodes[geno]] = ind.location[d]

    return coordinates


def splitInt16(int16):
    '''
    Take in a 16 bit integer, and return the top and bottom 8 bit integers

    Maybe not the most effecient? My best attempt based on my knowledge of python
    '''
    int16 = np.uint16(int16)
    bits = np.binary_repr(int16, 16)
    top = int(bits[:8], 2)
    bot = int(bits[8:], 2)
    return np.uint8(top), np.uint8(bot)


def GlueInt8(int8_t, int8_b):
    '''
    Take in 2 8-bit integers, and return the respective 16 bit integer created
    byt gluing the bit representations together

    Maybe not the most effecient? My best attempt based on my knowledge of python
    '''
    int8_t = np.uint8(int8_t)
    int8_b = np.uint8(int8_b)
    bits_a = np.binary_repr(int8_t, 8)
    bits_b = np.binary_repr(int8_b, 8)
    ret = int(bits_a + bits_b, 2)
    return np.uint16(ret)


def weighted_trees(ts, sample_weight_list, node_fun=sum):
    '''
    Here ``sample_weight_list`` is a list of lists of weights, each of the same
    length as the samples in the tree sequence ``ts``. This returns an iterator
    over the trees in ``ts`` that is identical to ``ts.trees()`` except that
    each tree ``t`` has the additional method `t.node_weights()` which returns
    an iterator over the "weights" for each node in the tree, in the same order
    as ``t.nodes()``.

    Each node has one weight, computed separately for each set of weights in
    ``sample_weight_list``. Each such weight is defined for a particular list
    of ``sample_weights`` recursively:

    1. First define ``all_weights[ts.samples()[j]] = sample_weights[j]``
        and ``all_weights[k] = 0`` otherwise.
    2. The weight for a node ``j`` with children ``u1, u2, ..., un`` is
        ``node_fun([all_weights[j], weight[u1], ..., weight[un]])``.

    For instance, if ``sample_weights`` is a vector of all ``1``s, and
    ``node_fun`` is ``sum``, then the weight for each node in each tree
    is the number of samples below it, equivalent to ``t.num_samples(j)``.

    To do this, we need to only recurse upwards from the parent of each
    added or removed edge, updating the weights.
    '''
    samples = ts.samples()
    num_weights = len(sample_weight_list)
    # make sure the provided initial weights lists match the number of samples
    for swl in sample_weight_list:
        assert(len(swl) == len(samples))

    # initialize the weights
    base_X = [[0.0 for _ in range(num_weights)] for _ in range(ts.num_nodes)]
    X = [[0.0 for _ in range(num_weights)] for _ in range(ts.num_nodes)]
    # print(samples)
    for j, u in enumerate(samples):
        for k in range(num_weights):
            X[u][k] = sample_weight_list[k][j]
            base_X[u][k] = sample_weight_list[k][j]

    z = zip(ts.trees(tracked_samples=ts.samples()), ts.edge_diffs())
    for t, (interval, records_out, records_in) in z:
        for edge in itertools.chain(records_out, records_in):
            u = edge.parent
            while u != msprime.NULL_NODE:
                for k in range(num_weights):
                    U = None
                    if(t.is_sample(u)):
                        U = [base_X[u][k]] + [X[u][k] for u in t.children(u)]
                    else:
                        U = [X[u][k] for u in t.children(u)]
                    X[u][k] = node_fun(U)
                u = t.parent(u)

        def the_node_weights(self):
            for u in self.nodes():
                yield X[u]

        # magic that uses "descriptor protocol"
        t.node_weights = the_node_weights.__get__(t, msprime.SparseTree)
        # t.node_weights = the_node_weights(t)
        yield t
