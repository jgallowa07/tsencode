
import msprime
import numpy as np
import sys
from weightedTrees import *
import pyslim

class tsEncoder():
    """
    This is a class which allows you to build up an 3D tensor encoding
    of an msprime TreeSequence, layer by layer. 

    When Visualized, This will take the first three / four layers to
    represent R,G,B, and A. 
    """    


    def __init__(self,
                treeSequence,
                width=None,         #max width for encoding
                height = None,      #max hieght for encoding
                dtype = None
                ):

        self.ts = treeSequence
        self.height = height
        if(height==None):
            self.height = treeSequence.num_nodes
        self.width = width
        if(width==None):
            self.width = int(treeSequence.sequence_length)
        self.datatype = dtype
        if(dtype==None):
            self.datatype = np.float64
        self.Encoding = None
        self.layerIndex = -1
 
    def initializeLayer(self):
        """ 
        initialize layer of ts.num_nodes X 
        """ 
        
        nn = self.ts.num_nodes
        layer = np.zeros([nn,int(self.width),1]).astype(self.datatype)
        if(self.layerIndex<0):
            self.Encoding = layer
            self.layerIndex = 0
        else:
            self.Encoding = np.append(self.Encoding,layer,axis=2)
            self.layerIndex += 1
        
        return None

    def addNodeTimeLayer(self):
        """
        (msprime TreeSequence, numpy dtype) -> None

        Add a layer to the Encoding which puts times on each
        node row.
        """
        
        self.initializeLayer()

        for i,node in enumerate(self.ts.nodes()):
            self.Encoding[i,0:self.width,self.layerIndex] = node.time
            
        return None

    def addParentPointer(self,split=False):
        """
        (msprime TreeSequence),bool -> None
        
        by adding adding all edges to the image,
        give each child a pointer to it's parent.

        if split parameter is True, Then it will split
        the parent pointer into two 8bit represenatations
        and add a layer for each
        """    

        self.initializeLayer()
        if(split):
            self.initializeLayer()

        for edge in self.ts.edges():
            child = edge.child
            left = int(edge.left)
            right = int(edge.right)
            if(self.width!=int(self.ts.sequence_length)):    
                left = int((left/self.ts.sequence_length)*self.width)
                right = int((right/self.ts.sequence_length)*self.width)
            if(split):
                top,bot = splitInt16(edge.parent)
                self.Encoding[child,left:right,self.layerIndex-1] = bot
                self.Encoding[child,left:right,self.layerIndex] = top          
            else:
                self.Encoding[child,left:right,self.layerIndex] = edge.parent
    
        return None

    def addBranchLengthLayer(self):
        '''
        Add a layer which will put branch length on each edge.
        '''
        
        self.initializeLayer()
        for edge in self.ts.edges():
            child = edge.child
            parent = edge.parent
            bl = self.ts.node(parent).time - self.ts.node(child).time
            left = int(edge.left)
            right = int(edge.right)
            if(self.width!=int(self.ts.sequence_length)):    
                left = int((left/self.ts.sequence_length)*self.width)
                right = int((right/self.ts.sequence_length)*self.width)
            self.Encoding[child,left:right,self.layerIndex] = bl

        return None

    def normalizeLayers(self,layers=[],scale=256):
        '''
        This function will normailize a layer by finding the
        max value in that layer, and normailizing all values 
        by putting them on the scale `scale`
        
        :param: layers should 
        '''

        for i in layers:
            fl = self.Encoding[:,:,i].flatten() 
            sh = self.Encoding[:,:,i].shape      
            ma = max(fl)
            nor = ((fl/ma)*scale)
            self.Encoding[:,:,i] = nor.reshape(sh)
            
        return None

    def addPropLayer(self):
        #TODO Finish making this universal
        """
        None -> None

        This will be a general propagation layer
        """

        pass

    def addSpatialPropLayer(self,function,dim=2,normalizePropWeights=True):
        """
        (msprime TreeSequence) -> None

        This will be a propagation layer which
        will use spatial locations as the initial weights
        
        The parameter, function, will be used to calculate
        weights of all parents as a function of their children's
        weights
        """

        #if(type(self.ts)!=pyslim.slim_tree_sequence.SlimTreeSequence):
        #    print(type(self.ts))
        #    print("must be a slim tree Sequence (I think)")
        #    return None
        
        for i in range(dim):
            self.initializeLayer()
    
        coordinates = []
        for i in range(dim):
            coordinates.append(np.zeros(self.ts.num_samples))

        for ind in self.ts.individuals():
            for d in range(dim):
                for geno in range(2):
                    coordinates[d][ind.nodes[geno]] = ind.location[d]
                    
        wt = weighted_trees(self.ts,coordinates,function)
        for t in wt:
            left = int((t.interval[0]/self.ts.sequence_length)*self.width)
            right = int((t.interval[1]/self.ts.sequence_length)*self.width)
            if(left == right):
                continue

            nodes = np.array([n for n in t.nodes()])
            inter = right-left
            weights = np.array([w for w in t.node_weights()])
            for i in range(dim):
                self.Encoding[nodes,left:right,self.layerIndex-i] = np.repeat(weights[:,i],inter).reshape([len(weights),inter])
    
        return None

    def addSpatialPropLayer(self,function,dim=2,normalizePropWeights=True):
        """
        (msprime TreeSequence) -> None

        This will be a propagation layer which
        will use spatial locations as the initial weights
        
        The parameter, function, will be used to calculate
        weights of all parents as a function of their children's
        weights
        """

        #if(type(self.ts)!=pyslim.slim_tree_sequence.SlimTreeSequence):
        #    print(type(self.ts))
        #    print("must be a slim tree Sequence (I think)")
        #    return None
        
        for i in range(dim):
            self.initializeLayer()
    
        coordinates = []
        for i in range(dim):
            coordinates.append(np.zeros(self.ts.num_samples))

        for ind in self.ts.individuals():
            for d in range(dim):
                for geno in range(2):
                    coordinates[d][ind.nodes[geno]] = ind.location[d]
                    
        wt = weighted_trees(self.ts,coordinates,function)
        for t in wt:
            left = int((t.interval[0]/self.ts.sequence_length)*self.width)
            right = int((t.interval[1]/self.ts.sequence_length)*self.width)
            if(left == right):
                continue

            nodes = np.array([n for n in t.nodes()])
            inter = right-left
            weights = np.array([w for w in t.node_weights()])
            for i in range(dim):
                self.Encoding[nodes,left:right,self.layerIndex-i] = np.repeat(weights[:,i],inter).reshape([len(weights),inter])
    
        return None

    def getEncoding(self,dtype=None):
        """
        return the actual encoding of the TreeSequence
        """
        
        if(dtype!=None):
            return self.Encoding.astype(dtype)
        else:
            return self.Encoding

    def visualize(self,saveas=None,show=True):
        from PIL import Image     

        imgArray = np.where(self.Encoding<0,0,self.Encoding).astype(np.uint8)[:,:,:3]
        img = Image.fromarray(imgArray,mode='RGB')
        if(show):
            img.show()
        if(saveas):
            img.save(saveas)
        
        #return imgArray
        

#------------END OF CLASS---------------
    
        
    
        
    
    
#-----------HELPER FUNCTIONS------------


def splitInt16(int16):
    '''
    Take in a 16 bit integer, and return the top and bottom 8 bit integers    

    Maybe not the most effecient? My best attempt based on my knowledge of python 
    '''
    int16 = np.uint16(int16)
    bits = np.binary_repr(int16,16)
    top = int(bits[:8],2)
    bot = int(bits[8:],2)
    return np.uint8(top),np.uint8(bot)

def GlueInt8(int8_t,int8_b):
    '''
    Take in 2 8-bit integers, and return the respective 16 bit integer created 
    byt gluing the bit representations together

    Maybe not the most effecient? My best attempt based on my knowledge of python 
    '''
    int8_t = np.uint8(int8_t)
    int8_b = np.uint8(int8_b)
    bits_a = np.binary_repr(int8_t,8)
    bits_b = np.binary_repr(int8_b,8)
    ret = int(bits_a+bits_b,2)
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
    #print(samples)
    for j, u in enumerate(samples):
        for k in range(num_weights):
            X[u][k] = sample_weight_list[k][j]
            base_X[u][k] = sample_weight_list[k][j]


    for t, (interval, records_out, records_in) in zip(ts.trees(tracked_samples=ts.samples()), ts.edge_diffs()):
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
        #t.node_weights = the_node_weights(t)
        yield t


