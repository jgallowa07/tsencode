from codecs import open as codecs_open
from setuptools import setup, find_packages
from warnings import warn


# Get the long description from the relevant file
with codecs_open('README.md', encoding='utf-8') as f:
    long_description = f.read()

HAVE_MSPRIME = False
try:
    import msprime
except ImportError:
    warn("`msprime` not present and must be installed")


setup(name='tsencode',
      version='0.1',
      description=u"Encode a tree sequence for visualization and ML purposes",
      long_description=long_description,
      classifiers=[],
      keywords='',
      author=u"Jared Galloway",
      author_email='jgallowa@uoregon.edu',
      url='https://github.com/jgallowa07/tsencode',
      license='MIT',
      packages=find_packages(exclude=[]),
      include_package_data=True,
      zip_safe=False,
      install_requires=['msprime>=0.7.0', 
                        'tskit', 
                        'pyslim',
                        'numpy',
                        'pillow'],
      extras_require={
          'dev': [],
      },
      setup_requires=[],
)
