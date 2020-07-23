
#print("jetson.inference.__init__.py")

# jetson.inference links against jetson.utils, and it needs loaded
import jetson.utils

# load jetson.inference extension module
from jetson_inference_python import *

VERSION = '1.0.0'
