import matplotlib.pyplot as plt
import numpy as np

vals = np.random.uniform(-1.,1.,1000000)
exp_vals = 2.0**vals
print np.mean(exp_vals)
print np.percentile(exp_vals,50.)

plt.figure()
plt.hist(exp_vals,bins=100)
plt.show()
