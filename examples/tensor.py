import apollo
import numpy as np
t = apollo.Tensor()
t.reshape((2,3))
t.set_values(2)
v = apollo.Tensor()
v.reshape((2,3))
v.set_values(5)
v += t
print v.get_mem()
v *= 3
print v.get_mem()
v *= t
print v.get_mem()
v.copy_from(t)
print v.get_mem()
c = v.get_mem()
c[:] = np.reshape(np.arange(v.count()), v.shape)
print v.get_mem()
v.set_mem(c)
