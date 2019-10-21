# TreeEncoding
This is the starting code for endcoding an [tskit](https://github.com/tskit-dev/tskit) TreeSequence Object into a 3D array for ML and visualization purposes



## Installation

To install `tsencode`, do
```
git clone https://github.com/jgallowa07/tsencode
cd tsencode
python3 setup.py install
```
You should also be able to install it with `pip install pyslim`. *NOT YET*
You'll also need an up-to-date [tskit](https://github.com/tskit-dev/tskit)

To run the tests to make sure everything is working, do:
```
python3 -m nose tests
```

*Note:* if you use `python3` you may need to replace `python` with `python3` above.

## Quickstart: Creating a quick visualization

1: There is a simple one-to-one encoding of a Tree Sequence that can be reached through `TsEncoder.add_one_to_one`

```
# Example 1
from tsencode import TsEncode
import msprime

ts = msprime.simulate(100,length=1e3,recombination_rate=1e-2)
encoder = TsEncoder(ts)
encoder.add_one_to_one()
encoder.normalize_layers(layers=[0])
encoder.visualize(show=True)
```

2: There are many possible uses of a Tree Sequence encoding, so, we'd like to have
an API where users can easily build visualizations with a given set of tools 
which could be easily extended in a framework.  

Here, I have mocked up a `TsEncoder` class which can be used to build up an encoding by adding 2D 'layers'
to a 3D tensor (numpy array). This will allow the user to "mix & match" the encoding properties which
most closely apply to thier specific problem. 

*This is very rough (aka no error handling, or many checks) but I wanted to post it so we can discuss the design*

Here's an example of how this work's so far:

```
# Example 2
import pyslim
import numpy as np
from tsencode import TsEncoder
from tsencode.helpers import get_genome_coordinates

ts = pyslim.load("tests/slim_trees/high_dispersal_2d_slim.trees")
ts = ts.recapitate(Ne=500,recombination_rate=1e-8)
ts = ts.simplify()
encoder = TsEncoder(ts,width=1000)
weights = get_genome_coordinates(ts, dim=2)
encoder.add_prop_layer(initial_weights=weights,function=np.mean)
encoder.add_node_time_layer()
encoder.normalize_layers([0,1,2],scale=256)
encoder.visualize(show=True)
```
