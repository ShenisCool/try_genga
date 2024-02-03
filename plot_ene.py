import pandas as pd
import matplotlib.pyplot as plt

#fig , axes_lst = plt.subplots(5,1)
fig , axes_lst = plt.subplots(2,1)
fig.suptitle("Energy change")

axe1=axes_lst[0]
axe2=axes_lst[1]
#axe3=axes_lst[2]
#axe4=axes_lst[3]
#axe5=axes_lst[4]

axe1.grid()
axe2.grid()
#axe3.grid()
#axe4.grid()
#axe5.grid()


axe1.set_title("L_rela-t")
axe2.set_title("E_rela-t")

axe1.set_xlabel('t(yrs)',fontdict={'size':16})
axe2.set_xlabel('t(yrs)',fontdict={'size':16})

#axe1.set_title("N-t")
#axe2.set_title("L_total-t")
#axe3.set_title("E_total-t")
#axe4.set_title("L_rela-t")
#axe5.set_title("E_rela-t")
	
filename='disk/Energytest.dat'

df = pd.read_csv(filename, sep='\s+')
#time0  N  V  T  LI  U  ETotal  LTotal  LRelativ  ERelativ
#time in years
#N: Number of particles
#V: Total potential energy , in
#T: Total Kinetic energy, in
#LI: Angular momentum lost at ejections, in
#U: Inner energy created from collisions, ejections or gas disk,
#ETotal: Total Energy, in
#LTotal: Total Angular Momentum, in
#LRelativ: (LTotal_t - LTotal_0)/LTotal_0, dimensionless
#ERelativ: (ETotal_t - ETotal_0)/ETotal_0, dimensionless

t=df.iloc[:,0]
N=df.iloc[:,1]
V=df.iloc[:,2]
T=df.iloc[:,3]

LI=df.iloc[:,4]
U=df.iloc[:,5]
E=df.iloc[:,6]
L=df.iloc[:,7]

LRE=df.iloc[:,8]
ERE=df.iloc[:,9]

print(t[1:35])

x1=t
x2=N
x3=L
x4=E
x5=LRE
x6=ERE

axe1.plot(x1,x5)
axe2.plot(x1,x6)

#axe1.plot(x1,x2)
#axe2.plot(x1,x3)
#axe3.plot(x1,x4)
#axe4.plot(x1,x5)
#axe5.plot(x1,x6)
#fig.set_figheight(10)
	
#axe1.set_ylim(0, 300) 

#plt.tight_layout()
plt.show()










