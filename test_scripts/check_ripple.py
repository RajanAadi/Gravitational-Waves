import ripplegw
import inspect
from ripplegw.waveforms.IMRPhenomD import gen_IMRPhenomD

print("Module path:", ripplegw.__file__)
print("Function signature of gen_IMRPhenomD:")
print(inspect.signature(gen_IMRPhenomD))
print("\nDocstring of gen_IMRPhenomD:")
print(gen_IMRPhenomD.__doc__)
