'''

Encoding Tree Sequence, 
This is code that takes in a treeSequence object from msprime,
and packs as much information into an array as possible. 

Author: Jared Galloway, Jerome Kelleher


'''
import msprime
import numpy as np
import sys
from PIL import Image

def DiscretiseTreeSequence(ts):
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
                      
    return tables.tree_sequence()

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

def EncodeTree_F32(ts,width=None):

    '''
    Encoding of a tree sequence into a matrix format ideally for DL,
    But also for visualization purposes
    
    for now let's try R = Time
                      G = Point to parent / Branch Length? 
                      B = Number of mutations? / type of mutations / total effect size?
    '''

    pic_width = ts.sequence_length
    if(width != None):  
        pic_width = width
                   
    A = np.zeros((ts.num_nodes,int(pic_width),3),dtype=np.float32) - 1
        
    for i,node in enumerate(ts.nodes()):
        A[i,0:pic_width,2] = np.float32(node.time)
        
    for edge in ts.edges():
        bl = ts.node(edge.parent).time - ts.node(edge.child).time
        child = edge.child
        parent = edge.parent
        left = int(edge.left)
        right = int(edge.right)
        if(width!=None):    
            left = int((left/ts.sequence_length)*width)
            right = int((right/ts.sequence_length)*width)
        A[child,left:right,0] = np.float32(parent)
        A[child,left:right,1] = np.float32(bl)

    return A


def EncodeTree_F64(ts,width=None):

    '''

    This one is for testing / visualization: 
    matches nodes.time being float64 

    Encoding of a tree sequence into a matrix format ideally for DL,
    But also for visualization purposes

    
    '''

    pic_width = ts.sequence_length
    if(width != None):  
        pic_width = width
                   
    A = np.zeros((ts.num_nodes,int(pic_width),3),dtype=np.float32) - 1
   
    for i,node in enumerate(ts.nodes()):
        #bl = ts.node(edge.parent).time - ts.node(edge.child).time
        A[i,0:int(ts.sequence_length),0] = node.time
        
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

    return A


def DecodeTree_F64(A): 
   
    '''
    Take in the array produced by 'EncodeTree()' and return a 
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


#Testing
if __name__ == "__main__": 
    
    '''
    #InverseTest
    test_dis_ts_en,ns = EncodeTree_F64(test_dis_ts)
    test_dis_ts_de = DecodeTree_F64(test_dis_ts_en,ns)

    for (i,(edge_b,edge_d)) in enumerate(zip(test_dis_ts.edges(),test_dis_ts_de.edges())):
        try:
            assert(edge_b == edge_d)
        except:
            print("MISMATCH, at edge number: ",i)
            print("before encode: ",edge_b)
            print("after decode: ",edge_d)
            print("--------")

    for (i,(node_b,node_d)) in enumerate(zip(test_dis_ts.nodes(),test_dis_ts_de.nodes())):
        try:
            assert(node_b == node_d)
        except:
            print("MISMATCH, at node number: ",i)
            print("before encode: ",node_b)
            print("after decode: ",node_d)
            print("--------")
    ''' 

    ts = msprime.simulate(10000,length=10000,recombination_rate=1e-5)
    #ts = msprime.simulate(50,length=100,random_seed=23)
    dts = DiscretiseTreeSequence(ts)
    ets = EncodeTree_F64(dts).astype(np.int8)
    #ets = EncodeTree_F32(dts,width=1000)

    img = Image.fromarray(ets,mode='RGB')
    img.save("test.png")
 






