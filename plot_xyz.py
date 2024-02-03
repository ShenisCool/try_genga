import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('expe/Outtest_p000000.dat', sep='\s+',header=None)

#with open('cpu/Outtest_p000216.dat', "r") as f:
#    data = f.read()
#print(data.shape)

#Output file Format: << t i m r x y z vx vy vz Sx Sy Sz amin amax emin emax aec aecT encc test a e inc O w M >>
t=df.iloc[:,0];
i=df.iloc[:,1];
m=df.iloc[:,2];
r=df.iloc[:,3]

x=df.iloc[:,4]
y=df.iloc[:,5]
z=df.iloc[:,6]

vx=df.iloc[:,7]
vy=df.iloc[:,8]
vz=df.iloc[:,9]

a=df.iloc[:,21]
e=df.iloc[:,22]
inc=df.iloc[:,23]
O=df.iloc[:,24]
w=df.iloc[:,25]
M=df.iloc[:,26]

x1=t
x2=e

#plt.xlim(0, 350) 
#plt.ylim(0, 1) 
#axe1.set_xlim(0, 300) 
#axe2.set_xlim(0, 300) 

font={'color':'black','weight':'normal','size':18}
plt.scatter(x1,x2,s=5,marker='.')
#plt.plot(x1,x2)
#plt.axis('equal')
plt.xlabel("X(AU)",fontdict=font,loc='center',labelpad=1.2)
plt.ylabel("Y(AU)",fontdict=font,loc='center',labelpad=1.2)
plt.grid()
plt.show()










