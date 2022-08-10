import matplotlib.pyplot as plt
import random

# 1000 tirages entre 0 et 150
x = [1, 2, 2, 3, 4, 4, 4, 4, 4, 5, 5]
plt.hist(x, range = (0, 5), bins = 5,
            edgecolor = 'red')


plt.xlabel('Mise')
plt.ylabel(u'Probabilité')
# plt.axis([0, 150, 0, 0.02])
plt.grid(True)
plt.show()