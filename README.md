# TreeEncoding
This is the starting code for endcoding an `tskit` TreeSequence Object into a 3D array for ML and visualization purposes



## Installation

To install `tsencode`, do
```
git clone https://github.com/jgallowa07/tsencode
cd tsencode
python setup.py install --user
```
You should also be able to install it with `pip install pyslim`. *NOT YET*
You'll also need an up-to-date [tskit](https://github.com/tskit-dev/tskit)

To run the tests to make sure everything is working, do:
```
python -m nose tests
```

*Note:* if you use `python3` you may need to replace `python` with `python3` above.

## Quickstart: Creating a quick visualization

There are two basic functions of tsencode at the moment. 

1: There is a simple one-to-one encoding of a Tree Sequence that can be reached through `tsencode.encode(ts)`

```
import tsencode
import msprime
from PIL import Image

ts = msprime.simulate(100,length=1e3,recombination_rate=1e-2)
enc = tsencode.encode(ts)
img = Image.fromarray(enc)
img.show()
```

2: Next, we have a `TsEncoder` class which can be used to build up an encoding by adding 2D 'layers'
to a 3D tensor (numpy array).

```
import tsencode
import msprime
from PIL import Image


```
