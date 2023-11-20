#Polar Contour plot
import numpy as np
import matplotlib.pyplot as plt
#generate r and theta arrays
rad_arr=np.radians(np.linspace(0,360,20))
r_arr=np.arange(0,1,.1)
#define function
def func(r,theta):
    return r*np.sin(theta)
r,theta=np.meshgrid(r_arr,rad_arr)
#get the values of respponse variables
values=func(r,theta)
#plot the polar coordinates
fig,ax=plt.subplots(subplot_kw=dict(projection='polar'))
ax.contourf(theta,r,values,cmp='Spectral_r')
plt.show()
