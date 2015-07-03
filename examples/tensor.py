import apollo
import numpy as np
t = apollo.Tensor()
t.reshape((2,3))
t.mem = 2
v = apollo.Tensor()
v.reshape((3,3))
v.mem = 5
v += t
print v.mem
v *= 3
print v.mem
v *= t
print v.mem
v.copy_from(t)
print v.mem
c = v.mem
c[:] = np.reshape(np.arange(v.count()), v.shape())
print v.mem
v.mem = c
