
import numpy as np
import matplotlib.pyplot as plt

from pylab import rcParams
rcParams['figure.figsize'] = 18, 8
rcParams['mathtext.default'] = 'regular'
rcParams['font.size'] = 20

t0 = 0.5
time = np.linspace(0, 1, 1000)


cusp = np.power(np.abs(time-t0), 0.6)
label_cusp = '$h(t_0 = 0.5) = 0.6$'
title_cusp = '$X(t) = |t-0.5|^{0.6} $'

label_osc = '$h(t_0 = 0.5) = 0.6$'
title_osc = '$X(t) = |t-0.5|^{0.6} \sin( |t-0.5|^{-0.5}  ) $'

osc  = np.power(np.abs(time-t0), 0.6)*np.sin( np.power(np.abs(time-t0), -0.5)  )


plt.figure()

plt.subplot(1,2,1)
plt.plot(time, cusp, linewidth=2, label = label_cusp)
plt.xlabel('$t$')
plt.ylabel('$X(t)$')
plt.title(title_cusp)
plt.legend()


plt.subplot(1,2,2)
plt.plot(time, osc, linewidth=2, label = label_osc)
plt.xlabel('$t$')
plt.ylabel('$X(t)$')
plt.title(title_osc)
plt.legend()

plt.show()