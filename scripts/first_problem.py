import numpy as np 
from scipy.optimize import curve_fit 
import matplotlib.pyplot as plt 
from typing import List

def model_function(x:np.ndarray, b:float, c:float) -> np.ndarray:
    return np.exp(x * -b) + c


x_data:np.ndarray = np.linspace(0, 10, num=100)
y_data:np.ndarray = np.exp(x_data * -0.5) + 2 + 0.1 * np.random.normal(scale =0.2, size=(100))

initial_guess:List[float] = [1.0, 1.0]
popt, pcov = curve_fit(model_function, x_data, y_data, p0=initial_guess)

print(f'the optimized values are for b : {popt[0]}, for c : {popt[1]}')


fig, ax = plt.subplots()
im = ax.imshow(pcov, cmap="coolwarm")

for i in range(pcov.shape[0]):
    for j in range(pcov.shape[1]):
        ax.text(i, j, "{:.5f}".format(pcov[i, j]))
plt.show()