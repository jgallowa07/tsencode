
from PIL import Image
#from TreesDirHelpers import *
from EncodeTreeSequence import *
from VisualizeTrees import *
import pyslim
from matplotlib import colors
from spatialTreeSequence import *
from tsEncoder import *


def shuffleNodes(x):
    t = np.arange(x.shape[0])
    np.random.shuffle(t)
    return x[t]

#Testing
if __name__ == "__main__": 

    #Test a treeSequence Encoding    
    #Note that with High Recombination, discretization will fail because it will 
    #make a left interval == right interval.
    #ts = msprime.simulate(50,Ne=1e4,length=1e3,recombination_rate=5e-10,random_seed=23)
    #test = TestEncodeTreeSequence(ts)
    #print(test)
    #sys.exit()

    #Do an Encoding and visualize it using PIL
    #ts = msprime.simulate(50,Ne=1e4,length=1e3,recombination_rate=5e-6,random_seed=23)
    #ts = msprime.simulate(50,Ne=1e4,length=1e3,recombination_rate=0,random_seed=23)
    #ts = pyslim.load("/Users/jaredgalloway/Documents/TreeSequenceEncoding/SLiMTrees/1830900922638_FIXED.trees")
    #ts = pyslim.load("/Users/jaredgalloway/Documents/TreeSequenceEncoding/SLiMTrees/neutral.trees")
    
    '''
    print(type(ts))
    if(type(ts) == pyslim.slim_tree_sequence.SlimTreeSequence):
        print("tru")

    for ind in ts.individuals():
        print(ind.nodes)

    ts = ts.simplify() 
    #sts = SpatialSlimTreeSequence(ts)

    node_x = np.zeros(ts.num_samples)
    node_y = np.zeros(ts.num_samples)
    for i in ts.individuals():
        loc_x = i.location[0]
        loc_y = i.location[1]
        node_x[i.nodes[0]] = loc_x
        node_x[i.nodes[1]] = loc_x
        node_y[i.nodes[0]] = loc_y
        node_y[i.nodes[1]] = loc_y
         
    #print(sample_node_locations)
    
    ets = EncodeTreeSequenceProp(ts,width=1000,return_8bit=True,sampleWeights=[node_x,node_y],propFunction=sum)

    #sys.exit()

    #ets = EncodeTreeSequence(ts,width=1000)
    #ets = shuffleNodes(ets)
    #print(ets.shape)

    
    img = Image.fromarray(ets,mode='RGB')
    img.save("high_dispersal_2d_sum.png")
    '''
    '''
    data = VisualizeNodes(treeSequence = ts,rescaled_time=True,RowsInImage=1000,ColumnsInImage=1000)
    #rgb = colors.hsv_to_rgb(data)

    img = Image.fromarray(data,mode='HSV')
    img.save("Viz.png")
    
    ''' 
    
    #ts = msprime.simulate(100,length=1e3,recombination_rate=1e-3)
    ts = pyslim.load("./SLiMTrees/low_dispersal.trees")
    ts = ts.simplify()
    encoder = tsEncoder(ts,width=1000)

    encoder.addParentPointer(split=True)
    #encoder.addNodeTimeLayer()
    #encoder.addBranchLengthLayer()
    encoder.addSpatialPropLayer(function=np.mean,dim=1)
    encoder.normalizeLayers([0,1,2],scale=256)

    
    #encoder.addParentPointer(split=True)
    encoding = encoder.getEncoding()
    print(encoding.shape)
    print(encoding.dtype)

    encoder.visualize(show=True)
    
    
    



