
import numpy as np
from PIL import Image

from .helpers import weighted_trees
from .helpers import splitInt16


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
            self.width = int(treeSequence.sequence_length)
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
        layer = np.zeros([nn, int(self.width), 1]).astype(self.datatype)
        if(self.layerIndex < 0):
            self.Encoding = layer
            self.layerIndex = 0
        else:
            self.Encoding = np.append(self.Encoding, layer, axis=2)
            self.layerIndex += 1

        return None

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
            left = int(edge.left)
            right = int(edge.right)
            if(self.width != int(self.ts.sequence_length)):
                left = int((left/self.ts.sequence_length)*self.width)
                right = int((right/self.ts.sequence_length)*self.width)
            if(split):
                top, bot = splitInt16(edge.parent)
                self.Encoding[child, left:right, self.layerIndex-1] = bot
                self.Encoding[child, left:right, self.layerIndex] = top
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
            left = int(edge.left)
            right = int(edge.right)
            if(self.width != int(self.ts.sequence_length)):
                left = int((left/self.ts.sequence_length)*self.width)
                right = int((right/self.ts.sequence_length)*self.width)
            self.Encoding[child, left:right, self.layerIndex] = bl

        return None

    def normalize_layers(self, layers=[], scale=256):
        '''
        This function will normailize a layer by finding the
        max value in that layer, and normailizing all values
        by putting them on the scale `scale`

        :param: layers should
        '''

        for i in layers:
            fl = self.Encoding[:, :, i].flatten()
            sh = self.Encoding[:, :, i].shape
            ma = max(fl)
            nor = ((fl/ma)*scale)
            self.Encoding[:, :, i] = nor.reshape(sh)

        return None

    def add_prop_layer(self):
        # TODO Finish making this universal
        """
        None -> None

        This will be a general propagation layer
        """

        pass

    def add_spatial_prop_layer(self, function, dim=2, normalizePropWeights=True):
        """
        (msprime TreeSequence) -> None

        This will be a propagation layer which
        will use spatial locations as the initial weights

        The parameter, function, will be used to calculate
        weights of all parents as a function of their children's
        weights
        """

        for i in range(dim):
            self.initialize_layer()
        coordinates = []
        for i in range(dim):
            coordinates.append(np.zeros(self.ts.num_samples))
        for ind in self.ts.individuals():
            for d in range(dim):
                for geno in range(2):
                    coordinates[d][ind.nodes[geno]] = ind.location[d]
        wt = weighted_trees(self.ts, coordinates, function)
        for t in wt:
            left = int((t.interval[0]/self.ts.sequence_length)*self.width)
            right = int((t.interval[1]/self.ts.sequence_length)*self.width)
            if left is right:
                continue
            nodes = np.array([n for n in t.nodes()])
            inter = right-left
            weights = np.array([w for w in t.node_weights()])
            for i in range(dim):
                layer = np.repeat(weights[:, i], inter).reshape([len(weights), inter])
                self.Encoding[nodes, left:right, self.layerIndex-i] = layer

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

        img_array = np.where(self.Encoding < 0, 0, self.Encoding).astype(np.uint8)
        img_array = img_array[:, :, :3]
        img = Image.fromarray(img_array, mode='RGB')
        if(show):
            img.show()
        if(saveas):
            img.save(saveas)
