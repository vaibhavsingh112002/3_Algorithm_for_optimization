# Practical 1 : Rectangular Contour Plot 
import numpy as np
import matplotlib.pyplot as plt
def func(x,y):
    return np.sin(x)**2+np.cos(y)**2
# generate 50 values between 0 & 5
x=np.linspace(0,5,50)
y=np.linspace(0,5,50)
# generate combination of grids
x,y=np.mgrid(x,y)
z=func(x,y)
#Draw rectangular contour plot
plt.contour(x,y,z,cmap='gist_rainbow_r');
plt.show()
