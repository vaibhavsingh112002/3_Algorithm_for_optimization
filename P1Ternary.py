
#Ternary Contour Plot
#install & import plotly
# pip install plotly
import numpy as np
import plotly.figure_factory as ff
# Define variable
a=np.array([0.,0.,0.,0.,1./3,1./3,1./3,2./3,2./3,1.])
b=np.array([0.,1./3,2./3,1.,0.,1./3,2./3,0.,1./3,0.])
c=1-a-b
#define function that generates response variable
func=(a-0.02)*b*(a-0.5)*(b-0.4)*(c-1)**2

#plot ternary contour
fig=ff.create_ternary_contour(np.array([a,b,c]),func,pole_labels=['a','b','c'],interp_mode='cartesian',colorscale='Viridis',)
fig.show()
#install pip install scikit-image
