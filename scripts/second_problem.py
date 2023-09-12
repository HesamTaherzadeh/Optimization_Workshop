import numpy as np 
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Tuple, List


# distance function 
def distance(x:Tuple, y:Tuple) -> float:
    return np.sqrt((x[0] - y[0])**2 + (x[1] - y[1])**2 + (x[2] - y[2])**2 )

# define model function / cost function 
def cost_function(points:Tuple, fixed_point:tuple) -> float:
    x1, y1, z1, x2, y2, z2 = points 
    cost = distance((x1, y1, z1), fixed_point) + distance((x2, y2, z2), fixed_point)
    return -1 * cost

def constraint(points:Tuple) -> List:
    x1, y1, z1, x2, y2, z2 = points 
    constraints_list = [
        x1**2 + y1**2 + z1**2 - 1,
        x2**2 + y2**2 + z2**2 - 1, 
        np.sqrt(distance((x1, y1, z1), (x2, y2, z2))) - 1,
        y1,
        y2
    ]
    return constraints_list


fixed_point = [0.1, 0.1, 1.0]

initial_guess = [0, 0, 1, 0, 0, 1]

bounds = (
    (-1, 1),
    (-1, 1),
    (-1, 1),
    (-1, 1),
    (-1, 1),
    (-1, 1),
)
    
cons = {
    'type' : 'eq',
    'fun' : constraint
}

result = minimize(cost_function, initial_guess, args=(fixed_point), bounds=bounds, constraints=cons, method='SLSQP').x

print(f"The optimized value for the p1 is [{result[0:3]},] and for p2 is [{result[3:]}]")
print(f"The distance should be equal to 1 : {distance(result[0:3], result[3:])}")



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, 2* np.pi, 100)
x = np.outer(np.sin(u), np.cos(v))
y = np.outer(np.sin(u), np.sin(v))
z = np.cos(v)[:, np.newaxis]

ax.plot_surface(x, y, z, color='b', alpha=0.3)

ax.scatter(*fixed_point, color='red', label= "Fixed point")

ax.scatter(*result[0:3], color='yellow', label= "$p_1$")
ax.scatter(*result[3:], color='black', label= "$p_2$")



plt.legend()
plt.show()
