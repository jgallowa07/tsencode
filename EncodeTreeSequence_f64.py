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

def EncodeTree(ts,width=None):

    '''
    Encoding of a tree sequence into a matrix format ideally for DL,
    But also for visualization purposes
    
    for now let's try R = Time of Parent (scaled to 256)
                      G = Branch Length (Scaled to 256)
                      #G = number of tracked samples
                      B = 
    '''
   
    oldest_time = max([node.time for node in ts.nodes()])
    A = np.zeros((ts.num_nodes,int(ts.sequence_length),3),dtype=np.float64) - 1
    
    for i,node in enumerate(ts.nodes()):
        #bl = ts.node(edge.parent).time - ts.node(edge.child).time
        A[i,0:int(ts.sequence_length),0] = node.time
        
    oldest_child_index = 0
    for edge in ts.edges():
        child = edge.child
        oldest_child_index = max(oldest_child_index,child)
        top,bot = splitInt16(edge.parent)
        A[edge.child,int(edge.left):int(edge.right),1] = top
        A[edge.child,int(edge.left):int(edge.right),2] = bot

    return A,ts.num_samples

def DecodeTree(A,numSamples): 
   
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
        if(row < numSamples):
            flag=1
        time = A[row,0,0]
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
    
    ts = msprime.simulate(100,length=1000,recombination_rate=1e-3)
    #ts = msprime.simulate(50,length=100,random_seed=23)
    test_dis_ts = DiscretiseTreeSequence(ts)
    
    test_dis_ts_en,ns = EncodeTree(test_dis_ts)
    test_dis_ts_de = DecodeTree(test_dis_ts_en,ns)

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
    
    A8 = test_dis_ts_en.astype(np.uint8) 
    img = Image.fromarray(A8,mode='RGB')
    img.save("test.png")

    






