#######################
参考信息
https://genga.readthedocs.io/en/latest/
https://bitbucket.org/sigrimm/genga/src/master/docs/index.rst?mode=view
#########################
本文件夹中：
grp.py可以生成xyz格式的初始粒子文件，可以根据需求修改生成需求
grpari.py可以生成aei格式的初始粒子文件，可以根据需求修改生成需求
plot_ene.py用于读取模拟生成的能量数据文件并绘图，可以修改文件以得到需要的表达效果
其中包括的数据有
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
plot_xyz.py用于读取模拟生成的能量数据文件并绘图，可以修改文件以得到需要的表达效果
其中包括的数据有（对于expe中的文件而言）<< t i m r x y z vx vy vz Sx Sy Sz amin amax emin emax aec aecT encc test a e inc O w M >>
#################################
cpu文件夹为原版genga的cpu版本，相对运行效率较低，但是程序比较好懂，也不用调整GPU的相关情况
source文件夹为原版genga的GP版本，运行效率较高，但是需要准备cuda、根据GPU情况在makefile中修改SM等，稍微有些繁琐
expe文件夹是经过修改的gpu版本genga，主要修改内容在gas.cu，output.cu,Host2.cu。
###############################
genga的简要运行逻辑（其他文件我们目前应该不是很需要关注）是：
define.h负责定义默认值等
param.dat负责输入运行参数、输入输出格式、输入输出文件、盘参数等
Host2.cu负责参数的整体调用、读取和输出
output.cu负责粒子状态等信息的输出和文件
每次修改后缀为.h或.cu的文件，必须重新编译make后方可运行./genga
###################################
以下部分仅讲解expe文件夹和修改内容的相关情况：
主要修改内容在gas.cu的GasAcc_Kernel中，这个函数负责计算气体对粒子的作用力
原版genga中这里包括gas_drag，gas_damping等效应，且使用模型相对简单。
我添加了包括type I migration,turbulence,two component disk，gap-openning mass，staller luminosity等模块。
还有一些内容有待测试和调整
########################################
如果需要添加自定义参数
例如要添加盘的性质参数，需要修改gas.cu的double Gas_parameter，要注意修改空间分配和赋值等；
在Host2.cu和define.h中进行相应的赋值，Host2.cu中进行参数提取er = fscanf (paramfile, "%lf", &P.G_alpha);要有对应输出则需要fprintf(infofile, "Gas disk inner edge: %g\n", P.G_rg0);（这里的P.G_rg0在gas.cu中被赋给了Gas_parameter）



















