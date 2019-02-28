from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use('fivethirtyeight')

#x = np.array([1, 2, 3, 4, 5, 6], dtype=np.float)
#y = np.array([5, 4, 6, 5, 6, 7], dtype=np.float)

def create_dataset(quant, variance, step=2, correlation=False):
    val = 1
    y = []
    for i in range(quant):
        yc = val + random.randrange(-variance, variance)
        y.append(yc)
        if correlation and correlation == 'pos':
            val+=step
        elif correlation and correlation =='neg':
            val-=step
    x = [i for i in range(len(y))]

    return np.array(x, dtype=np.float), np.array(y, dtype=np.float)


def best_fit_sope_intercept(x, y):
    m = ((mean(x) * mean(y)) - mean(x*y)) / ((mean(x)**2) - mean(x**2))
    b = mean(y) - m*mean(x)
    return m, b


def squared_error(y_orig, y_line):
    return sum((y_line - y_orig)**2)

def coefficient_of_determination(y_orig, y_line):
    y_mean_line = [mean(y_orig) for y in y_orig]
    squared_error_regr = squared_error(y_orig, y_line)
    squared_error_y_mean = squared_error(y_orig, y_mean_line)
    return 1 - (squared_error_regr/squared_error_y_mean)


x, y = create_dataset(80, 10, 2, correlation=False)

m, b = best_fit_sope_intercept(x, y)

regression_line = [(m*xc)+b for xc in x]

predict_x = 8
predict_y = m*predict_x + b

r_squared = coefficient_of_determination(y, regression_line)

print(r_squared)

plt.scatter(x, y)
plt.scatter(predict_x, predict_y, s=100, color="g")
plt.plot(x, regression_line)
plt.show()
