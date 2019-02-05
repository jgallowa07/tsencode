'''

Encoding Tree Sequence, 
This is code that takes in a treeSequence object from msprime,
and packs as much information into an array as possible. 

Author: Jared Galloway, Jerome Kelleher


'''
import msprime
import itertools
import numpy as np
import sys
from PIL import Image


def DiscretiseTreeSequence(ts):
    '''
    Disretise float values within a tree sequence
    also Put parent time on the scale of {0,256}. for colour purposes.
    '''    

    tables = ts.dump_tables()
    nodes = tables.nodes
    oldest_time = max(nodes.time)

    nodes.set_columns(flags=nodes.flags,
                      #time = (nodes.time/oldest_time)*256,
                      #time = np.around((nodes.time/oldest_time)*256),
                      #time = (nodes.time/oldest_time)*256,
                      time = np.arange(0,len(nodes)),
                      population = nodes.population)
                      
    return tables.tree_sequence()

def splitInt16(int16):
    '''
    Take in a 16 bit integer, and return the top and bottom 8 bit integers    
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
    #print("num_nodes: ",ts.num_nodes)
    #print("sequence_length: ",ts.sequence_length)
   
    #oldest_time = max([node.time for node in ts.nodes()])
    oldest_child_index = 0
    A = np.zeros((ts.num_nodes,int(ts.sequence_length),3),dtype=np.int8) - 1
    for edge in ts.edges():
        child = edge.child
        oldest_child_index = max(oldest_child_index,child)
        top,bot = splitInt16(edge.parent)
        #bl = ts.node(edge.parent).time - ts.node(edge.child).time
        #A[edge.child,int(edge.left):int(edge.right)+1,0] = (ts.node(edge.parent).time/oldest_time)*256
        A[edge.child,int(edge.left):int(edge.right)+1,0] = ts.node(edge.parent).time
        A[edge.child,int(edge.left):int(edge.right)+1,1] = top
        A[edge.child,int(edge.left):int(edge.right)+1,2] = bot

    #There is no information in rows that don't contain children
    return A[:oldest_child_index+1,:,:] 
    #return A

def DecodeTree(A,numSamples): 
   
    '''
    Take in the array produced by 'EncodeTree()' and return a 
    the inverse operation to produce a TreeSequence() for testing.
    
    '''

    num_rows = A.shape[0]    
    num_columns = A.shape[1]    
    print("A.shape: ",A.shape)

    tables = msprime.TableCollection(sequence_length=num_columns)
    node_table = tables.nodes
    edge_table = tables.edges

    for row in range(num_rows):
        flag=0
        if(row < numSamples):
            flag=1
        node_table.add_row(flags=flag,time=float(row))
    
        for column in range(num_columns):   
            top = A[row,column,1]
            bot = A[row,column,2]
            parent = GlueInt8(top,bot)
            edge_table.add_row(left=column,right=column+1,parent=parent,child=row)  
        
    for edge in edge_table:
        if(edge.parent >= len(node_table)):
            node_table.add_row(flags=0,time=float(parent))
        
    tables.simplify()
    ts = tables.tree_sequence()
             
    return ts
    #return None

#ts = msprime.simulate(5,length=10,recombination_rate=2.5e-3,random_seed=23)
ts = msprime.simulate(5,length=10,random_seed=23)
test_dis_ts = DiscretiseTreeSequence(ts)
test_dis_ts_en = EncodeTree(test_dis_ts)
test_dis_ts_de = DecodeTree(test_dis_ts_en,5)

for edge_b,edge_d in zip(test_dis_ts.edges(),test_dis_ts_de.edges()):
    assert(edge_b == edge_d)

#for node in tsd.nodes():
#    print(node)
#tsde = EncodeTree(tsd)
#print(tsde.shape)
#tsdd = DecodeTree(tsde,numSamples=10)

#DecodeTree()

#ET = EncodeTree(tsd)
#img = Image.fromarray(ET,mode='RGB')
#img.save("Recent4.png")


