'''

Encoding Tree Sequence,
This is code that takes in a treeSequence object from msprime,
and packs as much information into an array as possible.

Author: Jared Galloway, Jerome Kelleher


'''
import numpy as np

from .helpers import splitInt16


def encode(ts, width=None, return_8bit=True):
    '''
    This one is for testing / visualization:
    matches nodes.time being float64

    Encoding of a tree sequence into a matrix format ideally for DL,
    But also for visualization purposes
    '''

    oldest = max([node.time for node in ts.nodes()])
    pic_width = ts.sequence_length
    if width is not None:
        pic_width = width
    A = np.zeros((ts.num_nodes, int(pic_width), 3), dtype=np.float64) - 1
    for i, node in enumerate(ts.nodes()):
        time = (node.time / oldest) * 256
        A[i, 0:int(ts.sequence_length), 0] = time
    for edge in ts.edges():
        child = edge.child
        top, bot = splitInt16(edge.parent)
        left = int(edge.left)
        right = int(edge.right)
        if width is not None:
            left = int((left / ts.sequence_length) * width)
            right = int((right / ts.sequence_length) * width)
        A[child, left:right, 1] = top
        A[child, left:right, 2] = bot
    if return_8bit:
        A = np.uint8(A)
    return A
