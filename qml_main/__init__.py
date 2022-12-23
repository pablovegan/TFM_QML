from .qml_model import *
from .qml_optimizer import *
from .qml_results.__initi__ import *

class Fitter(object):
     def __init__(self,whatever):
         self.field1 = 0
         self.field2 = whatever

     # Imported methods

     # static methods need to be set
     from ._static_example import something
     something = staticmethod(something)

     # Some more small functions
     def printHi(self):
         print("Hello world")