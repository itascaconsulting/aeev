from numpy.distutils.core import setup
from numpy.distutils.misc_util import Configuration

def configuration(parent_package='', top_path=None):
    config = Configuration('aeev', parent_package, top_path)
    config.add_extension("aeev",
                          sources=["aeev.c", "makeop.h"],
                          include_dirs=['.'])

    return config

if __name__ == '__main__':
    setup(**configuration(top_path='').todict())

#from distutils.core import setup, Extension
#setup(ext_modules = [ Extension("aeev", sources=["aeev.c"])])
