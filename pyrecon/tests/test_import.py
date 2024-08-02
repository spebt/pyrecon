# from pyrecon import __version__
# print(__version__)
import os,sys
top_dir= os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(top_dir)
import pyrecon
print(pyrecon.__version__)
print(pyrecon.__all__)
