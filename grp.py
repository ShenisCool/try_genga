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
#Input file Format: << x y z m vx vy vz >>
#m默认 7.332687758e-09
#np.zeros(shape, dtype = float, order = 'C')

#data = np.ar

xsign=np.zeros(N)
ysign=np.zeros(N)
zsign=np.zeros(N)
vzsign=np.zeros(N)
ang=np.zeros(N)
a=np.zeros(N)

x=np.zeros(N)
y=np.zeros(N)
z=np.zeros(N)
m=np.zeros(N)
vx=np.zeros(N)
vy=np.zeros(N)
vz=np.zeros(N)

r=np.zeros(N)
v=np.zeros(N)

print(x)

m[0]=20*me
#x[0]=1.0
#y[0]=0.0
#xsign[0]=1
#ysign[0]=0
#z[0]=0

for i in range(0,N):
	#m[i]=mdefault#np.random.rand(1)*mrange+mmin
	#print(m[i])
	#np.random.seed()
	ang[i]=np.random.rand(1)*360	
	a[i]=5.2#np.random.rand(1)*arange+amin


	#xsign[i] = np.random.randint(0,2)*2-1
	#ysign[i] = np.random.randint(0,2)*2-1
	zsign[i] = np.random.randint(0,2)*2-1

	x[i]=a[i]*np.cos(ang[i])
	y[i]=a[i]*np.sin(ang[i])
	z[i]=zsign[i]*np.random.rand(1)*zrange
	
for i in range(0,N):
	xsign[i]=x[i]/np.abs(x[i])
	ysign[i]=y[i]/np.abs(y[i])
	
	r[i]=np.sqrt(x[i]*x[i]+y[i]*y[i]+z[i]*z[i])
	v[i]=1/np.sqrt(r[i])
	vx[i]=-ysign[i]*np.abs(v[i]/r[i]*y[i])
	vy[i]=+xsign[i]*np.abs(v[i]/r[i]*x[i])
	vz[i]=vzsign[i]*np.random.rand(1)*vzrange
#

#vx[0]=0
data = np.transpose(np.vstack((x,y,z,m,vx,vy,vz)))
np.savetxt('gaslikeB.dat', data, fmt='%.9e', delimiter=' ')



