'''

Encoding Tree Sequence, 
This is code that takes in a treeSequence object from msprime,
and packs as much information into an array as possible. 

Author: Jared Galloway, Jerome Kelleher


'''
import msprime
import numpy as np
import sys
from weightedTrees import *

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

def EncodeTreeSequence(ts,width=None,return_8bit=True):

    '''
    This one is for testing / visualization: 
    matches nodes.time being float64 

    Encoding of a tree sequence into a matrix format ideally for DL,
    But also for visualization purposes    
    '''
    oldest = max([node.time for node in ts.nodes()])

    pic_width = ts.sequence_length
    if(width != None):  
        pic_width = width
                   
    A = np.zeros((ts.num_nodes,int(pic_width),3),dtype=np.float64) - 1
   
    for i,node in enumerate(ts.nodes()):
        time =(node.time/oldest)*256
        A[i,0:int(ts.sequence_length),0] = time
    
    for edge in ts.edges():
        child = edge.child
        top,bot = splitInt16(edge.parent)
        left = int(edge.left)
        right = int(edge.right)
        if(width!=None):    
            left = int((left/ts.sequence_length)*width)
            right = int((right/ts.sequence_length)*width)
        A[edge.child,left:right,1] = top
        A[edge.child,left:right,2] = bot
    
    if(return_8bit):
        A = np.uint8(A)

    return A

def EncodeTreeSequenceProp(ts,
                        width=None,
                        return_8bit=True,
                        sampleWeights=None,
                        propFunction=None,
                        normalizePropWeights=True):

    '''
    This one is for testing / visualization: 
    matches nodes.time being float64 

    Encoding of a tree sequence into a matrix format ideally for DL,
    But also for visualization purposes    
    '''
    oldest = max([node.time for node in ts.nodes()])

    pic_width = ts.sequence_length
    if(width != None):  
        pic_width = width
                   
    A = np.zeros((ts.num_nodes,int(pic_width),3),dtype=np.float64) - 1
   
    #for i,node in enumerate(ts.nodes()):
    #    time =(node.time/oldest)*256
    #    A[i,0:int(ts.sequence_length),0] = time
    
    for edge in ts.edges():
        child = edge.child
        top,bot = splitInt16(edge.parent)
        left = int(edge.left)
        right = int(edge.right)
        if(width!=None):    
            left = int((left/ts.sequence_length)*width)
            right = int((right/ts.sequence_length)*width)
        #A[child,left:right,1] = top
        A[child,left:right,2] = bot
    
    wt = weighted_trees(ts, sampleWeights, propFunction)
    for t in wt:
        left = int((t.interval[0]/ts.sequence_length)*width)
        right = int((t.interval[1]/ts.sequence_length)*width)
        if(left == right):
            continue
        nodes = np.array([n for n in t.nodes()])
        inter = right-left

        weights = np.array([w for w in t.node_weights()])
        for i in range(2):
            A[nodes,left:right,i] = np.repeat(weights[:,i],inter).reshape([len(weights),inter])

        #for col in range(left,right):
        #    A[nodes,col,0] = weights

        #for idx,node in enumerate(t.nodes()):
        #    A[node,left:right,0] = weights[idx]
    
    if(normalizePropWeights):
        for i in range(2):
            fl = A[:,:,i].flatten() 
            sh = A[:,:,i].shape      
            ma = max(fl)
            nor = ((fl/ma)*256)
            A[:,:,i] = nor.reshape(sh)

 
    if(return_8bit):
        A = np.where(A<0,0,A).astype(np.uint8)

    return A

#InverseTest
def TestEncodeTreeSequence(ts):
    '''
    This is the inverse test for the EncodeTreeSequence() function.
    
    it simply encodes a discretized tree, then compares it to it's
    un-encoded counterpart

    msprime TreeSequence -> boolean
    '''

    dts = DiscretizeTreeSequence(ts)
    edts = EncodeTreeSequence(dts,return_8bit=False)
    de_dts = DecodeTree(edts)

    flag = True

    for (i,(edge_b,edge_d)) in enumerate(zip(dts.edges(),de_dts.edges())):
        try:
            assert(edge_b == edge_d)
        except:
            flag=False
            print("MISMATCH, at edge number: ",i)
            print("before encode: ",edge_b)
            print("after decode: ",edge_d)
            print("--------")

    for (i,(node_b,node_d)) in enumerate(zip(dts.nodes(),de_dts.nodes())):
        try:
            assert(node_b == node_d)
        except:
            flag=False
            print("MISMATCH, at node number: ",i)
            print("before encode: ",node_b)
            print("after decode: ",node_d)
            print("--------")

    return flag

def DecodeTree(A): 
   
    '''
    Take in the array produced by 'EncodeTreeSequence()' and return a 
    the inverse operation to produce a TreeSequence() for testing.
    
    '''

    num_rows = A.shape[0]    
    num_columns = A.shape[1]    

    tables = msprime.TableCollection(sequence_length=num_columns)
    node_table = tables.nodes
    edge_table = tables.edges
    pop_table = tables.populations
    pop_table.add_row()

    for row in range(num_rows):

        flag=0
        time = A[row,0,0]
        if(time == 0.0):
            flag=1
        node_table.add_row(flags=flag,time=float(time),population=0)
    
        for column in range(num_columns):   
            
            top = A[row,column,1]
            bot = A[row,column,2]
            #for padding, we don't add edges 
            if((top < 0) | (bot < 0)):  
                continue
            parent = GlueInt8(top,bot)
            edge_table.add_row(left=column,right=column+1,parent=parent,child=row)  
    

    tables.sort()        
    tables.simplify()
    ts = tables.tree_sequence()
             
    return ts

def DiscretizeTreeSequence(ts):
    '''
    Disretise float values within a tree sequence
    
    mainly for testing purposes to make sure the decoding is equal to pre-encoding.
    '''    

    tables = ts.dump_tables()
    nodes = tables.nodes
    edges = tables.edges
    oldest_time = max(nodes.time)

    nodes.set_columns(flags=nodes.flags,
                      time = (nodes.time/oldest_time)*256,
                      population = nodes.population
                        )
    edges.set_columns(left = np.round(edges.left),
                      right = np.round(edges.right),
                      child = edges.child,
                      parent = edges.parent
                        )
    ret = None
    try:
        ret = tables.tree_sequence()
    except:
        print("msprime doesn't like the discretization, prob too much recombination")

    return ret

