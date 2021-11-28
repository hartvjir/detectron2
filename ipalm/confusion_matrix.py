from matplotlib import cm
from matplotlib.pyplot import subplots, show
from numpy import linspace, bincount, hstack, cumsum
from numpy.random import rand, random_integers
import numpy as np



print(__doc__)

import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

material_probs = [0.0, 0.1, 0.5, 0.2, 0.1, 0.1]
random_probs = np.random.uniform(size=(10,100))
print(random_probs)



y_actu = [2, 0, 2, 2, 0, 1, 1, 2, 2, 0, 1, 2]
y_pred = [0, 0, 2, 1, 0, 2, 1, 0, 2, 0, 2, 2]

print(confusion_matrix(y_actu, y_pred))

