
import numpy as np
from PIL import Image

from tsencode.helpers import weighted_trees
from tsencode.helpers import splitInt16


class TsEncoder():
    """
    This is a class which allows you to build up an 3D tensor encoding
    of an msprime TreeSequence, layer by layer.

    When Visualized, This will take the first three / four layers to
    represent R,G,B, and A.
    """

    def __init__(self,
                 treeSequence,
                 width=None,         # max width for encoding
                 height=None,      # max hieght for encoding
                 dtype=None):

        self.ts = treeSequence
        self.height = height
        if height is None:
            self.height = treeSequence.num_nodes
        self.width = width
        if width is None:
            self.width = treeSequence.sequence_length
        self.width = int(self.width)
        self.datatype = dtype
        if dtype is None:
            self.datatype = np.float64
        self.Encoding = None
        self.layerIndex = -1

    def initialize_layer(self):
        """
        initialize layer of ts.num_nodes X
        """

        nn = self.ts.num_nodes
        layer = np.zeros([nn, int(self.width), 1]).astype(self.datatype) - 1
        if(self.layerIndex < 0):
            self.Encoding = layer
            self.layerIndex = 0
        else:
            self.Encoding = np.append(self.Encoding, layer, axis=2)
            self.layerIndex += 1

        return None

    def map_locus_to_column(self, locus):
        return int((locus / self.ts.sequence_length) * self.width)

    def add_node_time_layer(self):
        """
        (msprime TreeSequence, numpy dtype) -> None

        Add a layer to the Encoding which puts times on each
        node row.
        """

        self.initialize_layer()

        for i, node in enumerate(self.ts.nodes()):
            self.Encoding[i, 0:self.width, self.layerIndex] = node.time

        return None

    def add_parent_pointer(self, split=False):
        """
        (msprime TreeSequence),bool -> None

        by adding adding all edges to the image,
        give each child a pointer to it's parent.

        if split parameter is True, Then it will split
        the parent pointer into two 8bit represenatations
        and add a layer for each
        """

        self.initialize_layer()
        if(split):
            self.initialize_layer()

        for edge in self.ts.edges():
            child = edge.child
            left = self.map_locus_to_column(edge.left)
            right = self.map_locus_to_column(edge.right)
            if(split):
                top, bot = splitInt16(edge.parent)
                self.Encoding[child, left:right, self.layerIndex - 1] = top
                self.Encoding[child, left:right, self.layerIndex] = bot
            else:
                self.Encoding[child, left:right, self.layerIndex] = edge.parent

        return None

    def add_branch_length_layer(self):
        '''
        Add a layer which will put branch length on each edge.
        '''

        self.initialize_layer()
        for edge in self.ts.edges():
            child = edge.child
            parent = edge.parent
            bl = self.ts.node(parent).time - self.ts.node(child).time
            left = self.map_locus_to_column(edge.left)
            right = self.map_locus_to_column(edge.right)
            self.Encoding[child, left:right, self.layerIndex] = bl

        return None

    def normalize_layers(self, layers=[], scale=256, trans="linear"):
        # TODO make log scale norm
        '''
        This function will normailize a layer by finding the
        max value in that layer, and normailizing all values
        by putting them on the scale `scale`

        :param: layers should
        '''

        for i in layers:
            fl = self.Encoding[:, :, i].flatten()
            sh = self.Encoding[:, :, i].shape
            if trans == "linear":
                ma = max(fl)
                nor = ((fl / ma) * scale)
            else:
                # This still needs work: Talk to Peter
                log_fl = np.log(fl + 1)
                ma = max(log_fl)
                nor = ((log_fl / ma) * scale)
            self.Encoding[:, :, i] = nor.reshape(sh)

        return None

    def add_prop_layer(self, initial_weights, function):
        # TODO Finish making this universal
        """
        None -> None

        This will be a general propagation layer
        """
        dim = len(initial_weights)
        for i in range(dim):
            self.initialize_layer()

            # check to see that each set of weights provided is equal to the number
            # of samples, If not throw an error.
            if self.ts.num_samples is not len(initial_weights[i]):
                # TODO raise error
                return None

        wt = weighted_trees(self.ts, initial_weights, function)
        for t in wt:
            left = self.map_locus_to_column(t.interval[0])
            right = self.map_locus_to_column(t.interval[1])

            # If the image has been squished enough, skip this tree because it
            # won't add anything to the encoding.
            if left is right:
                continue

            nodes = np.array([n for n in t.nodes()])
            inter = right - left
            weights = np.array([w for w in t.node_weights()])
            l, r = left, right
            n, i, w = nodes, inter, weights
            for d in range(dim):
                pl = self.layerIndex - d
                # this may be confusing, but essentially
                # it's just a vectorized way to populate the encoding with the node weights # NOQA
                self.Encoding[n, l:r, pl] = np.repeat(w[:, (dim-1)-d], i).reshape([len(w), i]) # NOQA

                # how I was originally doing it
                # self.Encoding[nodes, left:right, self.layerIndex-i] = np.repeat(weights[:, i], inter).reshape([len(weights), inter])    # NOQA

        return None

    def add_one_to_one(self):
        """
        This function should replicate the one-to-one function found int
        tsencode/one_to_one.py.

        For now, it is mostly for testing purposes do to code that has been
        written to inverse this function.
        """

        self.add_node_time_layer()
        self.add_parent_pointer(split=True)

        return None

    def get_encoding(self, dtype=None):
        """
        return the actual encoding of the TreeSequence
        """

        if dtype is not None:
            return self.Encoding.astype(dtype)
        else:
            return self.Encoding

    def visualize(self, saveas=None, show=True):

        # TODO: not sure that zero'ing out neg numbers is the right heuristic here
        img_array = np.where(self.Encoding < 0, 0, self.Encoding).astype(np.uint8)

        # if there is less than three layers, add trivial layers to the image
        if(self.layerIndex < 2):
            nn = self.ts.num_nodes
            trivial_layers = np.zeros([nn, int(self.width), 2 - self.layerIndex]).astype(np.uint8) # NOQA
            img_array = np.append(img_array, trivial_layers, axis=2)
        else:
            img_array = img_array[:, :, :3]
        img = Image.fromarray(img_array, mode='RGB')
        if(show):
            img.show()
        if(saveas):
            img.save(saveas)
