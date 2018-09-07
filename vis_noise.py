import numpy as np
import matplotlib.pyplot as plt
from colored_noise import powerlaw_psd_gaussian


from pylab import rcParams
rcParams['figure.figsize'] = 20, 5
rcParams['mathtext.default'] = 'regular'
rcParams['font.size'] = 20




noise_1 = powerlaw_psd_gaussian(1.0, 4096)
noise_2 = powerlaw_psd_gaussian(3.0, 4096)



plt.figure()
plt.subplot(1,2,1)
plt.plot(noise_1)
plt.xlabel('time')
plt.grid()
plt.title('Less regular')
plt.subplot(1,2,2)
plt.plot(noise_2)
plt.xlabel('time')
plt.grid()
plt.title('More regular')
plt.show()


plt.tight_layout()