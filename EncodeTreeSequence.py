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

    print("argsort: ",)

    nodes.set_columns(flags=nodes.flags,
                      #time = (nodes.time/oldest_time)*256,
                      #time = np.around((nodes.time/oldest_time)*256),
                      #time = (nodes.time/oldest_time)*256,
                      time = np.arange(0,len(nodes)),
                      #np.argsort(nodes.time))
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
   
    #oldest_time = max([node.time for node in ts.nodes()])
    oldest_child_index = 0
    A = np.zeros((ts.num_nodes,int(ts.sequence_length),3),dtype=np.int8) - 1
    for edge in ts.edges():
        child = edge.child
        oldest_child_index = max(oldest_child_index,child)
        top,bot = splitInt16(edge.parent)
        #bl = ts.node(edge.parent).time - ts.node(edge.child).time
        #A[edge.child,int(edge.left):int(edge.right),0] = ts.node(edge.parent).time
        A[edge.child,int(edge.left):int(edge.right),0] = (ts.node(edge.parent).time/oldest_time)*256
        A[edge.child,int(edge.left):int(edge.right),1] = top
        A[edge.child,int(edge.left):int(edge.right),2] = bot

    #There is no information in rows that don't contain children
    #return A[:oldest_child_index+1,:,:] 

    #But, we'll return the whole thing for simplicity
    return A,ts.num_samples,oldest_child_index

def DecodeTree(A,numSamples,oldestChild): 
   
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

    for row in range(oldestChild+1):
        flag=0
        if(row < numSamples):
            flag=1
        node_table.add_row(flags=flag,time=float(row),population=0)
    
        for column in range(num_columns):   
            
            top = A[row,column,1]
            bot = A[row,column,2]
            
            #for padding, we don't add edges 
            if((top < 0) | (bot < 0)):  
                continue
    
            parent = GlueInt8(top,bot)

            edge_table.add_row(left=column,right=column+1,parent=parent,child=row)  
        
    #Adam and Eve; for all the parents that didn't act as a child as well
    for edge in edge_table:
        if(edge.parent >= len(node_table)):
            node_table.add_row(flags=0,time=float(parent),population=0)

    tables.sort()        
    tables.simplify()
    ts = tables.tree_sequence()
             
    return ts

#Testing
if __name__ == "__main__": 
    #A = np.array([1, 2, 5, 2, 1, 25, 2,1])
    #print(A)
    #print(np.argsort(A).astype(float)) 
    
    ts = msprime.simulate(50,length=100,random_seed=23,recombination_rate=1e-3)
    #ts = msprime.simulate(50,length=100,random_seed=23)
    test_dis_ts = DiscretiseTreeSequence(ts)
    test_dis_ts_en,ns,oc = EncodeTree(test_dis_ts)
    
    test_dis_ts_de = DecodeTree(test_dis_ts_en,ns,oc)

    for (i,(edge_b,edge_d)) in enumerate(zip(test_dis_ts.edges(),test_dis_ts_de.edges())):
        try:
            assert(edge_b == edge_d)
        except:
            print("MISMATCH, at edge number: ",i)
            print("before encode: ",edge_b)
            print("after decode: ",edge_d)
            print("--------")
            #sys.exit()

    for (i,(node_b,node_d)) in enumerate(zip(test_dis_ts.nodes(),test_dis_ts_de.nodes())):
        try:
            assert(node_b == node_d)
        except:
            print("MISMATCH, at node number: ",i)
            print("before encode: ",node_b)
            print("after decode: ",node_d)
            print("--------")
            #sys.exit()
    

    #assert(test_dis_ts == test_dis_ts_de)


    #for node in tsd.nodes():
    #    print(node)
    #tsde = EncodeTree(tsd)
    #print(tsde.shape)
    #tsdd = DecodeTree(tsde,numSamples=10)

    #DecodeTree()

    #ET = EncodeTree(tsd)
    #img = Image.fromarray(ET,mode='RGB')
    #img.save("Recent4.png")
    

