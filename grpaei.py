import numpy as np
import pandas as pd


#xrange,yrange,zrange

N=1
#xrange=5
mmin=0
mrange=1.0e-10
mdefault=1.0e-12
mj=1.0e-03
me=mj/300
aj=1.0
ae=1.0

amin=4.5
arange=1
#ymin=0.2
zrange=1.0e-04
vzrange=1.0e-07

xsign=0
ysign=0
zsign=0
vzsign=0
a=0
#Input file Format: << a(or P) e inc O w M(orT) m >>
#m默认 7.332687758e-09
#np.zeros(shape, dtype = float, order = 'C')

#data = np.ar

a=np.zeros(N)
e=np.zeros(N)
inc=np.zeros(N)
O=np.zeros(N)
w=np.zeros(N)
M=np.zeros(N)
m=np.zeros(N)

m[0]=20*me

for i in range(0,N):
	e[i]=0.3
	m[i]=10*me#np.random.rand(1)*mrange+mmin
	np.random.seed()
	O[i]=np.random.rand(1)*360	
	w[i]=np.random.rand(1)*360
	M[i]=np.random.rand(1)*360


#for i in range(0,N):
	
#

#vx[0]=0
data = np.transpose(np.vstack((a,e,inc,O,w,M,m)))
np.savetxt('gaslikeaei.dat', data, fmt='%.9e', delimiter=' ')



