
from PIL import Image
from EncodeTreeSequence import *

#Testing
if __name__ == "__main__": 

    #Test a treeSequence Encoding    
    #Note that with High Recombination, discretization will fail because it will 
    #make a left interval == right interval.
    ts = msprime.simulate(50,Ne=1e4,length=1e3,recombination_rate=5e-10,random_seed=23)
    test = TestEncodeTreeSequence(ts)
    print(test)
    
    #Do an Encoding and visualize it using PIL
    ts = msprime.simulate(50,Ne=1e4,length=1e3,recombination_rate=5e-7,random_seed=23)
    ets = EncodeTreeSequence(ts)
    img = Image.fromarray(ets,mode='RGB')
    img.save("test.png")
    





