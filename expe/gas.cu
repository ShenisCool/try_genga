#include "Orbit2.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>

#define facrho 1.0/sqrt(2.0 * M_PI)
#define PI 3.1415926535897932
#define e 2.7182818284590452353602874713526624977572

double *Gas_rg_d;
double *Gas_zg_d;
double *Gas_rho_d;
double3 *GasDisk_d;
double3 *GasAcc_d;


double Gas_parameters_h[12];
__constant__ double Gas_parameters_c[12];

int G_Nr_g, G_Nr_p;

__host__ void Data::GasAlloc(){

	G_Nr_g = (P.G_rg1 - P.G_rg0 + 0.5 * P.G_drg) / P.G_drg - 1;
	G_Nr_p = (P.G_rp1 - P.G_rg0 + 0.5 * P.G_drg) / P.G_drg + 1;

	//printf("Gas N %d %d\n", G_Nr_g, G_Nr_p);

	cudaMalloc((void **) &Gas_rg_d, G_Nr_g * sizeof(double));
	cudaMalloc((void **) &Gas_zg_d, G_Nr_g * def_Gasnz_g * sizeof(double));
	cudaMalloc((void **) &Gas_rho_d, G_Nr_g * def_Gasnz_g * sizeof(double));
	cudaMalloc((void **) &GasDisk_d, G_Nr_p * sizeof(double3));
	cudaMalloc((void **) &GasAcc_d, G_Nr_p * def_Gasnz_p * sizeof(double3));

#if def_CPU ==1
	Gas_rg_h = (double*)malloc(G_Nr_g * sizeof(double));
	Gas_zg_h = (double*)malloc(G_Nr_g * def_Gasnz_g * sizeof(double));
	Gas_rho_h = (double*)malloc(G_Nr_g * def_Gasnz_g * sizeof(double));
	GasDisk_h = (double3*)malloc(G_Nr_p * sizeof(double3));
	GasAcc_h = (double3*)malloc(G_Nr_p * def_Gasnz_p * sizeof(double3));
#endif

	Gas_parameters_h[0] = P.G_dTau_diss;
	Gas_parameters_h[1] = P.G_alpha;
	Gas_parameters_h[2] = P.G_beta;
	Gas_parameters_h[3] = P.G_Sigma_10;
	Gas_parameters_h[4] = P.G_Mgiant;
	Gas_parameters_h[5] = P.G_rg0;
	Gas_parameters_h[6] = P.G_rg1;
	Gas_parameters_h[7] = P.G_drg;
	Gas_parameters_h[8] = P.G_turstralpha;
	Gas_parameters_h[9] = P.G_accrate_0;
	Gas_parameters_h[10] = P.G_L_s0;
	Gas_parameters_h[11] = P.G_diskvis_alpha;

#if def_CPU == 0
	cudaMemcpyToSymbol(Gas_parameters_c, Gas_parameters_h, 12 * sizeof(double), 0, cudaMemcpyHostToDevice);
#else
#endif
}

//__global__ void exp(double *a, double *b,double *c){
//	c=pow(a,b);//c=1;想办法自定义exp，或者pow(e,x)//但仍需实现rand
//}
__global__ void cudaRand(double *d_out)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    curandState state;
    curand_init((unsigned long long)clock() + i, 0, 0, &state);

    d_out[i] = curand_uniform_double(&state);
}
// *************************************************
// This function corresponds to the msrGasTable function in the file master.c in pkdgrav_planets.
// 20世纪的老模型了，我们用新的模型结构可以直接改在这里
//
// ****************************************************
__global__ void GasDisk_kernel(double *Gas_rg_d, double *Gas_zg_d, double *Gas_rho_d, const int G_Nr_g){

	int ig = blockIdx.x * blockDim.x + threadIdx.x; // r
	int jg = blockIdx.y * blockDim.y + threadIdx.y; // z

	double G_alpha = Gas_parameters_c[1];
	double G_beta =  Gas_parameters_c[2];
	double G_Sigma10 = Gas_parameters_c[3];
	double rg0 = Gas_parameters_c[5];	//inner edge of the gas disk
	double drg = Gas_parameters_c[7];	//spacing of the gas disk

//	double rin, ro;
//	if(uniform != 1){
//		rin = 0.1 + dTime/(2.0 * M_PI * dTau_diss); // time scale for the inner edge to move 1AU
//	}

	if(ig < G_Nr_g){
		double rg = rg0 + drg * (ig + 0.5);	//staggered grid

		double h = def_h_1 * rg * pow(rg, G_beta); //beta = 0.25 comes from Temperature profile
		double Sigma = G_Sigma10 * pow(rg, -G_alpha);

//if(jg == 0) printf("GasDisk %d %g %g %g\n", ig, rg, h, Sigma);	
//			if(uniform != 1){
//				ro = rg + 0.5 * drg; // radius of the outer cel boundary
//				if(rin > ro){
//					Sigma = 0.0;
//				}
//				else if(fabs(rin - rg) < 0.5 * drg){
//					//if((ro - rin) > drg) send error
//					Sigma *= (ro - rin)/drg;
//				}
//			}
		if(jg < def_Gasnz_g){
			if(jg == 0){
				Gas_rg_d[ig] = rg;
//printf("Gas_rg %d %g\n", ig, Gas_rg_d[ig]);
			}

			double zg = 0.01 * (0.5 + jg) * rg;
			double zh = zg / h;
			Gas_zg_d[ig * def_Gasnz_g + jg] = zg;
			Gas_rho_d[ig * def_Gasnz_g + jg] = facrho * (Sigma / h) * exp(-0.5 * zh * zh);
//printf("Gas_rho %g %g %g\n", rg, zg, Gas_rho_d[ig * def_Gasnz_g + jg]);
		}
	}
}

// **********************************************
// First Kind elliptic integral
// This function corresponds to the rf function in the file master.c in pkdgrav_planets.
// It is based on numerical recipes and the paper from Carlson 
// **********************************************
__device__ double rf(double x, double y, double z){
	double lambda, mu, imu, X, Y, Z;
	double E2, E3, sqrtx, sqrty, sqrtz;
	double errtol = 0.008;
	double third = 1.0/3.0;

	double xt, yt, zt;

	xt = x;
	yt = y;
	zt = z;

//	if(fmin(fmin(x, y), z) < 0.0 || fmin(fmin(x + y, x + z), y + z) < 1.5e-38 || fmax(fmax(x, y), z) > 3.0e37){
//		printf("invalid arguments in first elliptical integral.");
//	}

	for(int i = 0; i < 10000; ++i){
		sqrtx = sqrt(xt);	
		sqrty = sqrt(yt);
		sqrtz = sqrt(zt);

		lambda = sqrtx * (sqrty + sqrtz) + sqrty * sqrtz;

		xt = 0.25 * (xt + lambda);
		yt = 0.25 * (yt + lambda);
		zt = 0.25 * (zt + lambda);

		mu = third * (xt + yt + zt);
		imu = 1.0 / mu;
		X = (mu - xt) * imu;
		Y = (mu - yt) * imu;
		Z = (mu - zt) * imu;

		if(fmax(fmax(fabs(X), fabs(Y)), fabs(Z)) <= errtol) break;
	}


	E2 = X * Y - Z * Z;
	E3 = X * Y * Z;

	return (1.0 + E2*(E2/24.0 - E3 * 3.0/44.0 - 0.1) + E3/14.0) / sqrt(mu);
}


// **********************************************
// Second Kind elliptic integral
// This function corresponds to the rf function in the file master.c in pkdgrav_planets.
// It is based on numerical recipes and the paper from Carlson 
// **********************************************
__device__ double rd(double x, double y, double z){
	double lambda, mu, imu, X, Y, Z;
	double EA, EB, EC, ED, EE, sqrtx, sqrty, sqrtz;
	double errtol = 0.008;
	double sum = 0.0;
	double fac = 1.0;

	double xt, yt, zt;

	xt = x;
	yt = y;
	zt = z;

//	if(fmin(x, y) < 0.0 || fmin(x + y, z) < 1.0e-25 || fmax(fmax(x, y), z) > 4.5e21){
//		printf("invalid arguments in second elliptical integral.");
//	}

	for(int i = 0; i < 10000; ++i){
		sqrtx = sqrt(xt);
		sqrty = sqrt(yt);
		sqrtz = sqrt(zt);

		lambda = sqrtx * (sqrty + sqrtz) + sqrty * sqrtz;
		sum += fac / (sqrtz * (zt + lambda));	//difference from master.c and numerical recipes
//sum += fac / (sqrtz * zt + lambda);
		fac *= 0.25;
		xt = 0.25 * (xt + lambda);
		yt = 0.25 * (yt + lambda);
		zt = 0.25 * (zt + lambda);

		mu = 0.2 * (xt + yt + 3.0 * zt);
		imu = 1.0 / mu;
		X = (mu - xt) * imu;
		Y = (mu - yt) * imu;
		Z = (mu - zt) * imu;

		if(fmax(fmax(fabs(X), fabs(Y)), fabs(Z)) <= errtol) break;
	}

	EA = X * Y;
	EB = Z * Z;
	EC = EA - EB;
	ED = EA - 6.0 * EB;
	EE = ED + EC + EC;


	return 3.0 * sum + fac * (1.0 + ED * (ED * 9.0/88.0 - Z * EE * 4.5/26.0 - 3.0/14.0) + Z * (EE / 6.0 + Z *(EC * -9.0/22.0 + Z * EA * 3.0/26.0))) /(mu * sqrt(mu));
}


// *************************************************
// This function corresponds to the msrGasTable function in the file master.c in pkdgrav_planets.
//
// ****************************************************
__global__ void gasTable_kernel(double *Gas_rg_d, double *Gas_zg_d, double *Gas_rho_d, double3 *GasDisk_d, double3 *GasAcc_d, const int G_Nr_g, const int G_Nr_p){
	
	int ip = blockIdx.x * blockDim.x + threadIdx.x; // r
	int jp = blockIdx.y * blockDim.y + threadIdx.y; // z
	//printf("ipXjp:%d    %d\n",ip,jp);
	volatile double ar, az;
	double rp, zp;
	double ellf, elle;

	double G_alpha = Gas_parameters_c[1];
	double G_beta =  Gas_parameters_c[2];
	double G_Sigma10 = Gas_parameters_c[3];
	double rg0 = Gas_parameters_c[5];	//inner edge of the gas disk
	double drg = Gas_parameters_c[7];	//spacing of the gas disk

	if(ip < G_Nr_p){	
		rp = rg0 + drg * ip;

		double h = def_h_1 * rp * pow(rp, G_beta);
		double Sigma = G_Sigma10 * pow(rp, -G_alpha);
//		if(uniform != 1){
//			ro = rp + 0.5 * drg; // radius of the outer cel boundary
//			if(rin > rp + 0.5 * drg){
//				Sigma = 0.0;
//			}
//			else if(fabs(rin - rp) < 0.5 * drg){
//				//if((ro - rin) > drg) send error
//				Sigma *= (ro - rin)/drg;
//			}
//		}

		if(jp == 0){
			GasDisk_d[ip].x = Sigma;
			GasDisk_d[ip].y = h;
			GasDisk_d[ip].z = rp;
			//printf("GasDisk_d %d %g %g %g\n", ip, rp, h, Sigma);
		}
		
	}
	__syncthreads();
	
	
	
	if(ip < G_Nr_p){
		rp = rg0 + drg * ip;
		if(jp < def_Gasnz_p){

			zp = (0.03 * jp) * rp;
			ar = 0.0;
			az = 0.0;
			for(int ig = 0; ig < G_Nr_g; ++ig){
				double rgas = Gas_rg_d[ig];
				double dzg = 0.03 * rgas;
				
				for(int jg = 0; jg < def_Gasnz_g; ++jg){

					double zgas = Gas_zg_d[ig * def_Gasnz_g + jg];
					double rho_gas = Gas_rho_d[ig * def_Gasnz_g + jg];

					volatile double rpzm = (zp - zgas) * (zp - zgas);
					volatile double rmzm = rpzm + (rp - rgas) * (rp - rgas);
					rpzm += (rp + rgas) * (rp + rgas);

					double k2 = (4.0 * rp * rgas) / rpzm;
					ellf = rf(0.0, 1.0 - k2, 1.0);
					elle = rf(0.0, 1.0 - k2, 1.0) - k2 * rd(0.0, 1.0 - k2, 1.0) / 3.0;
					volatile double temp = -2.0 * (rho_gas / sqrt(rpzm)) * rgas * drg * dzg;
					elle /= rmzm;

					ar += (temp/rp) * (elle * (rp * rp - rgas * rgas - (zp - zgas) * (zp - zgas)) + ellf);
					az += 2.0 * temp * (zp - zgas) * elle;

					zgas = -zgas;
					rpzm = (zp - zgas) * (zp - zgas);
					rmzm = rpzm + (rp - rgas) * (rp - rgas);
					rpzm += (rp + rgas) * (rp + rgas);

					k2 = (4.0 * rp * rgas) / rpzm;
					ellf = rf(0.0, 1.0 - k2, 1.0);
					elle = rf(0.0, 1.0 - k2, 1.0) - k2 * rd(0.0, 1.0 - k2, 1.0) / 3.0;
			
					temp = -2.0 * (rho_gas / sqrt(rpzm)) * rgas * drg * dzg;
					elle /= rmzm;

					ar += (temp/rp) * (elle * (rp * rp - rgas * rgas - (zp - zgas) * (zp - zgas)) + ellf);
					az += 2.0 * temp * (zp - zgas) * elle;

				}
			}
			GasAcc_d[ip * def_Gasnz_p + jp].x = ar;
			GasAcc_d[ip * def_Gasnz_p + jp].y = az;
			GasAcc_d[ip * def_Gasnz_p + jp].z = zp;
//	printf("GasAcc %d %d %g %g %g\n", ip, jp, ar, az, zp);
		}
	}
}

__host__ int Data::setGasDisk(){
	cudaError_t error;
	dim3 NTGasDisk(128, 1, 1);
	dim3 NBGasDisk((G_Nr_g + 127) / 128, def_Gasnz_g, 1);
	GasDisk_kernel <<< NBGasDisk, NTGasDisk >>> (Gas_rg_d, Gas_zg_d, Gas_rho_d, G_Nr_g);

	dim3 NTGasTabel(128, 1, 1);
	dim3 NBGasTabel((G_Nr_p + 127) / 128, def_Gasnz_p, 1);

	gasTable_kernel <<< NBGasTabel, NTGasTabel >>>(Gas_rg_d, Gas_zg_d, Gas_rho_d, GasDisk_d, GasAcc_d, G_Nr_g, G_Nr_p);
	cudaDeviceSynchronize();

	error = cudaGetLastError();
	fprintf(masterfile, "Gas Table error = %d = %s\n",error, cudaGetErrorString(error));
	if(error != 0){
		printf("Gas Table error = %d = %s\n",error, cudaGetErrorString(error));
		return 0;
	}
	else return 1;
}

// *************************************************
// This kernel corresponds to the pkdGasAccel function in the file pkd.c in pkdgrav_planets.
//
// ****************************************************
__global__ void GasAcc_kernel(double4 *x4_d, double4 *v4_d, int *index_d, double3 *GasDisk_d, double3 *GasAcc_d, double *time_d, double2 *Msun_d, double *dt_d, int N, double *EnergySum_d, int Nst, double Ct, int UsegasPotential, int UsegasEnhance, int UsegasDrag, int UsegasTidalDamping, double Mgiant, int Nstart, int G_Nr_p,int a){

	int idy = threadIdx.x;
	int id = blockIdx.x * blockDim.x + idy + Nstart;

	double G_dTau_diss = Gas_parameters_c[0];
	double G_alpha =     Gas_parameters_c[1];
	double G_beta =      Gas_parameters_c[2];
	double G_Sigma10 =   Gas_parameters_c[3];
	double rg0 =         Gas_parameters_c[5];	//inner edge of the gas disk
	double rg1 =         Gas_parameters_c[6];	//outher edge of the gas disk
	double drg =         Gas_parameters_c[7];       //spacing of the gas disk
	double alpha_t =     Gas_parameters_c[8];
	double mdot_gas =    Gas_parameters_c[9];
	double L_s0 =        Gas_parameters_c[10];
	double alpha_vis =   Gas_parameters_c[11];
	//printf("%g,%g,%g\n\n\n",mdot_gas,L_s0,alpha_vis);

	if(id < N + Nstart){
		
		int st = 0;
		if(Nst > 1) st = index_d[id] / def_MaxIndex;
		double dt = dt_d[st] * Ct;
		double dTime = time_d[st] / 365.25;

		double Msun = Msun_d[st].x;
		double U = 0.0;

		double3 v_rel3;
		double v_rel_r, v_rel_th;
		double big = 1.0e7;

		double depfac = exp(-dTime/(G_dTau_diss));

		double4 x4 = x4_d[id];
		double4 v4 = v4_d[id];

		double r1 = x4.x * x4.x + x4.y * x4.y;
		double rsq = r1 + x4.z * x4.z;
		double vsq = v4.x * v4.x + v4.y * v4.y + v4.z * v4.z;
		r1 = sqrt(r1);

//r1 = id * 0.01;
		double r = sqrt(rsq);

		volatile double a_r = 0.0;
		volatile double a_th = 0.0;
		volatile double a_x = 0.0;
		volatile double a_y = 0.0;
		volatile double a_z = 0.0;
		
		double h = def_h_1 * r1 * pow(r1, G_beta);//原版高
		double hgas=h/r1;       //gas disk aspect ratio 
		double Sigma = G_Sigma10 * pow(r1, -G_alpha);//原版sigma
		//printf("%g\n",30.0*pow((h/0.05),2.5)*pow((G_alpha/0.001),0.5)*(Msun/def_StarK2));

	
		if(r1 > rg0 && r1 < rg1 && x4.z/r1 < 1.5){ //otherwise there is no gas
		
			//****************************
			//gap opening mass //alpha_t = 0.001 default, turbulent viscos para
			//*********************************
			//double mgo=30.0*pow((hgas/0.05),2.5)*pow((alpha_t/0.001),0.5)*(Msun/def_StarK2);//earth mass as before, I havent change it
			double mgo=5.8*pow((hgas/0.05),2.5)*pow((alpha_t/0.001),0.5)*(Msun/def_StarK2);
			//***********************************
			//two component disk
			//*************************************
			double M_s=Msun/def_StarK2;
			double L_s=L_s0;
			double ll0;
			//L_s0=1;
			//lumi
			//printf("%g\n",dTime);//dTime[yrs],time_d[day]
			double t0=365.25*1e6;
			double t1=pow(10,2.024)*365.25*1e6-365.25*1e6;
			double tn=365.25*1e6;//原本t0 = 365.25*1d6 tn = 365.25d0*1d6 ! day
			if(0){//条件是一个选项，但是无所谓
				if(time_d[st]<t1){
					L_s=L_s0*pow(((time_d[st]+365.25*1e6)/tn),-0.5975);
					L_s=pow(((time_d[st]+365.25*1e6)/tn),-0.5975)*L_s0;
				}else{
					ll0=log10(L_s0)+0.2452;
					L_s=pow(10.0,ll0-1.454334595);
				}
			}else{
				L_s=pow(Msun/def_StarK2,2);
			}
			//**************************
			//two component
			//**************************************
			//gas viscous timescale
			double p,q;
			//对于例子而言，恒星0.1,alpha_t=1e-3,吸积=1e-8,但是单位是yr-1注意会不会换算，盘后期有所衰减
			double kap0=0.01;
			double kap=kap0;//κ=κ0 (T_g/1 K)
			p=-15.0/14.0;
			q=2.0/7.0;
			double Omega = pow((Msun/pow(r1,3)),0.5) ;
			double hgas0_irr=(2.45e-2)*pow(L_s/1.0,1.0/7.0)*pow(M_s/1.0,-4.0/7.0);
			double hgas_irr = hgas0_irr*pow(r,q);
			double v_0=alpha_vis*(hgas_irr,2)*Omega;
			double ts_0=1.0/v_0/(3.0*pow(p+2,2));
			//viscous heated region
			p=-0.375;
			q=-1.0/16.0;
			double siggas0_vis=740*pow(mdot_gas/1.0e-8,1.0/2.0)*pow(M_s/1.0,1.0/8.0)*pow(alpha_vis/1e-3,-3.0/4.0)*pow(kap/1e-2,-1.0/4.0);
			double siggas_vis=siggas0_vis*pow(r,p);
			double hgas0_vis=4.5e-2*pow(mdot_gas/1e-8,1.0/4.0)*pow(M_s/1.0,-5.0/16.0)*pow(alpha_vis/1e-3,-1.0/8.0)*pow(kap/1e-2,1.0/8.0);
			double hgas_vis=hgas0_vis*pow(r,q);
			double T0_vis=500.0*pow(mdot_gas/1e-8,1.0/2.0)*pow(M_s/1.0,3.0/8.0)*pow(alpha_vis/1e-3,-1.0/4.0)*pow(kap/1e-2,1.0/4.0);
			double T_vis=T0_vis*pow(r,2*q-1);
			//stellar irradition region
			p = - 15.0/14.0;
			q = 2.0/7.0;
			//double siggas0_irr=2500;
			double siggas0_irr = 2500*(mdot_gas/1e-8)*pow(L_s/1.0,-2.0/7.0)*pow(M_s/1.0,9.0/14.0)*pow(alpha_vis/1e-3,-1);
			double siggas_irr = siggas0_irr*pow(r,p);
			hgas0_irr = 2.45e-2*pow(L_s/1.0,1.0/7.0)*pow(M_s/1.0,-4.0/7.0);
			hgas_irr = hgas0_irr*pow(r,q);
			//temperature  at 1AU and as a function of r  in K
			double T0_irr = 150.0*pow(L_s/1.0,2.0/7.0)*pow(M_s/1.0,-1.0/7.0);
			double T_irr = T0_irr*pow(r,2*q-1);
			//transition radius for two regions
			double rtrans = pow(500.0/150.0,56.0/39.0)*pow(mdot_gas/1e-8,28.0/39.0)*pow(M_s/1.0,29.0/39.0)*(alpha_vis/1e-3,-14.0/39.0)*pow(L_s/1.0,-16.0/39.0)*pow(kap/1e-2,14.0/39.0);
			//double rtrans = 60; 
			double fs; 
			//m_gap = gap_opening(hgas,alpha_t,mstar) ! in earth mass 
			//m_planet = mpl/K2/EARTH_MASS ! planet mass in earth mass  
			double m_pl=x4.w/def_StarK2*1.3e6;
		
			if(1){//if 1 include the inner viscously heated region
				fs = 1.0/(1.0+pow((m_pl/mgo),4)) ;
			}else{
				fs=0.0;
			}
			double siggas = fs*siggas_vis+(1.0-fs)*siggas_irr;
			double Temp = fs*T_vis+(1.0-fs)*T_irr;
			double hgas = fs*hgas_vis+(1.0-fs)*hgas_irr;
			
			//midplane volume density  in g/cm^3
			double rhogas_mid = siggas/pow(2*PI,0.5)/hgas/def_AU;
			//gas volume density rho(r,z)
			double rhogas = rhogas_mid*pow(e,-pow(x4.z,2)/pow(hgas*r,2)); // in g/cm^3
			//gas headwind prefactor at 1AU
			double etagas0 = (2.0-p-q)/2.0;
			//gas headwind prefactor
			double etagas = etagas0*pow(hgas,2);
			
			//gas surface density at r  [in solar mass*K2]
			siggas = siggas/def_Solarmass*pow(def_AU*100,2)*def_StarK2; //in solar mass*K2，这里原文MSUN*AU**2*K2，注意是单位换算
			siggas_vis = siggas_vis/def_Solarmass*pow(def_AU*100,2)*def_StarK2;  //in solar mass*K2
			siggas_irr = siggas_irr/def_Solarmass*pow(def_AU*100,2)*def_StarK2; //in solar mass*K2
			//printf("SIG:%g,%g,%g,%g\n",Sigma,siggas,siggas_irr,siggas_vis);
			//printf("h:%g,%g,%g,%g\n",h,hgas,hgas0_vis,hgas0_irr);
			if(1){
				Sigma=siggas;
			}
			//double etagas = 1; //linshide
			//****************************
			//pebble isolation mass
			//*********************************
			//we need turbulent viscos alpha, headwind prefactor,stokes number……
			//double alpha_t = 0.001;
			double f_miso = pow((h/0.05),3)*(0.34*pow(log10(0.001)/log10(alpha_t),4)+0.66);
			f_miso =f_miso*(1.0-(2.5-2.0*etagas/pow(h,2))/6.0);
			double mpi = 25.0*f_miso;//pebble isolation based on Bitsch+2018
			mpi = (30*pow((alpha_t/0.001),0.5)*pow((h/0.05),2.5)/2.3)*(Msun/def_StarK2);
			//原文pebble_iso = 25d0*f_miso !+ alpha/(2d0*ts)/(4.76d-3/f_miso)
			//pebble_iso = pebble_iso*(mstar/K2) ! consider the stellar mass dependence  
			//pebble_iso = 30d0*(alpha_t/1d-3)**0.5*(hgas/5d-2)**2.5/2.3 ! in earth mass for 1 solar mass star
			//pebble_iso = pebble_iso*(mstar/K2) ! consider the stellar mass dependence, in earth mass 
			//"h" in this line meas aspect ratio, not the height, maybe height at 1AU equal

			Sigma *= depfac;//depfac = exp(-dTime/(G_dTau_diss))
//if(id < 100) printf("%d %g %g %g\n", id, r1, Sigma, h);

			int ip = floor((r1 - rg0) / drg) + 1;

			double zh = x4.z / h;

			double m = x4.w;
			if(m == 0.0) m = def_MgasSmall;
			
			//gas potential

			if(UsegasPotential == 1 || (UsegasPotential == 2 && m < Mgiant)){

				if(ip >= G_Nr_p){	//outside GasDisk Table
					a_r += -2.0 * M_PI * Sigma;	
					a_z += a_r * erf(zh);
					if(G_alpha == 2.0) a_r *= log(r1 / drg);
				}
				else if(ip > 0){
					double rr0 = GasDisk_d[ip - 1].z;
					double rr1 = GasDisk_d[ip].z;

					int jp0 = floor(fabs(x4.z)/(rr0 * 0.03));
					int jp1 = floor(fabs(x4.z)/(rr1 * 0.03));

					double zz00 = GasAcc_d[(ip - 1) * def_Gasnz_p + jp0].z;
					double zz01 = GasAcc_d[(ip - 1) * def_Gasnz_p + jp0 + 1].z;
					double zz10 = GasAcc_d[ip * def_Gasnz_p + jp1].z;
					double zz11 = GasAcc_d[ip * def_Gasnz_p + jp1 + 1].z;

					double dr00 = __dmul_rn((x4.z - zz00) , (x4.z - zz00)) + __dmul_rn((r1 - rr0) , (r1 - rr0));
					double dr01 = __dmul_rn((x4.z - zz01) , (x4.z - zz01)) + __dmul_rn((r1 - rr0) , (r1 - rr0));
					double dr10 = __dmul_rn((x4.z - zz10) , (x4.z - zz10)) + __dmul_rn((r1 - rr1) , (r1 - rr1));
					double dr11 = __dmul_rn((x4.z - zz11) , (x4.z - zz11)) + __dmul_rn((r1 - rr1) , (r1 - rr1));
					
					dr00 = fmin(1.0/sqrt(dr00), big);
					dr01 = fmin(1.0/sqrt(dr01), big);
					dr10 = fmin(1.0/sqrt(dr10), big);
					dr11 = fmin(1.0/sqrt(dr11), big);
					double drtotal = dr00 + dr01 + dr10 + dr11;


					double a_r_t0 = GasAcc_d[(ip - 1) * def_Gasnz_p + jp0].x * dr00 + GasAcc_d[(ip - 1) * def_Gasnz_p + jp0 + 1].x * dr01 + GasAcc_d[ip * def_Gasnz_p + jp1].x * dr10 + GasAcc_d[ip * def_Gasnz_p + jp1 + 1].x * dr11;
					double a_z_t0 = GasAcc_d[(ip - 1) * def_Gasnz_p + jp0].y * dr00 + GasAcc_d[(ip - 1) * def_Gasnz_p + jp0 + 1].y * dr01 + GasAcc_d[ip * def_Gasnz_p + jp1].y * dr10 + GasAcc_d[ip * def_Gasnz_p + jp1 + 1].y * dr11;
					a_r_t0 /= drtotal;
					a_z_t0 /= drtotal;

					a_r += a_r_t0 * depfac;
					if(x4.z >= 0.0){
						a_z += a_z_t0 * depfac;
					}
					else a_z -= a_z_t0 * depfac;

				}

//printf("%d %g %g %g %g %g | %g\n", id, r1, Sigma, h, a_r, a_z, erf(zh));
			}


			if(Sigma > 0.0){
				
				//Enhanced Drag
				double Soft;
				if(UsegasEnhance == 1 || (UsegasEnhance == 2 && m < Mgiant)){
					
					if(m < def_M_Enhance && m > def_MgasSmall){
						double pid = log(m/def_fMass_min) / log(def_M_Enhance/def_fMass_min);
						double jc = def_M_Enhance/def_Mass_pl;
						m = pow(jc, pid);
						m *= def_Mass_pl;
						Soft = v4.w * pow(m/x4.w, 1.0/3.0);
					}
					else{
					
						Soft = v4.w;
					}
				}
				else{
					Soft = v4.w;

				}
				
				//vKep
				double v_kep = 0.0;
				if(UsegasDrag > 0 || UsegasTidalDamping > 0){
					v_kep = sqrt(Msun * def_ksq / r1 - a_r * r1); 
				}
		
				//gas drag
				if(UsegasDrag == 1 || (UsegasDrag == 2 && m < Mgiant)){
					double eta = 0.5 * ((G_alpha + 1.75) * h * h + 0.5 * x4.z * x4.z) / (r1 * r1);
					double v_gas = v_kep * (1.0 - eta);	//Change that to v_kep * sqrt(1.0 - 2.0 * eta);

					double rho = facrho * Sigma / h * exp(-0.5 * zh * zh);
					v_rel3.x = v4.x + v_gas * x4.y / r1; 
					v_rel3.y = v4.y - v_gas * x4.x / r1;
					v_rel3.z = v4.z;

					v_rel_r = (x4.x * v_rel3.x + x4.y * v_rel3.y) / r1;
					v_rel_th = (x4.x * v_rel3.y - x4.y * v_rel3.x) / r1;
					
					double v_rel = v_rel3.x * v_rel3.x + v_rel3.y * v_rel3.y + v_rel3.z * v_rel3.z;
					v_rel = sqrt(v_rel);
					if(m > 0.0) v_rel *= M_PI / (2.0 * m) * def_Gas_cd * Soft * Soft * rho;
					else v_rel = 0.0;

					a_x += -v_rel * v_rel3.x;
					a_y += -v_rel * v_rel3.y;
					a_z += -v_rel * v_rel3.z;
				}

				//tidal damping
				if(UsegasTidalDamping == 1 || (UsegasTidalDamping == 2 && m < Mgiant)){
					v_rel3.x = v4.x + v_kep * x4.y / r1;
					v_rel3.y = v4.y - v_kep * x4.x / r1;
					v_rel3.z = v4.z;
					v_rel_r = (x4.x * v_rel3.x + x4.y * v_rel3.y) / r1;
					v_rel_th = (x4.x * v_rel3.y - x4.y * v_rel3.x) / r1;

					double Mtot = Msun + m;
					double Etot = 0.5 * vsq - def_ksq * Mtot / r;
					double3 L;
					L.x = x4.y * v4.z - x4.z * v4.y;
					L.y = x4.z * v4.x - x4.x * v4.z;
					L.z = x4.x * v4.y - x4.y * v4.x;

					double Lsq = L.x * L.x + L.y * L.y + L.z * L.z;
					double esq = 1.0 + 2.0 * Etot * Lsq /(Mtot * Mtot);
					if(esq < 0.0) esq = 0.0;
					double isq = 1.0 - L.z * L.z / Lsq;
					double a = -0.5/Etot;

					double r1h2 = r1 * r1 / (h * h);
					
					double chi = 0.5 * a * sqrt(esq + isq) / h;
					double chi3 = chi * chi * chi;

					double iTau_tid1 = m * (Sigma * r1 * r1) * r1h2	/ (r1 * sqrt(r1)); //a = ri because of circular orbit
					double iTauwave_gwave = iTau_tid1 * r1h2 / (1.0 + 0.25 * chi3);
					iTau_tid1 *= 3.8 * (1.0 - 0.683 * chi3 * chi) / (1.0 + 0.269329 * chi3 * chi * chi);

					//type I migration
					a_th += -0.5 * iTau_tid1 * v_kep; //circular

					//damping
					a_r += (0.104 * v_rel_th + 0.176 * v_rel_r) * iTauwave_gwave;
					a_th += (-1.736 * v_rel_th + 0.325 * v_rel_r) * iTauwave_gwave;
					a_z += (-1.088 * v_rel3.z - 0.871 * v_kep * x4.z / r1) * iTauwave_gwave;
				}
			}
			//genga自带migration
			//a_x += (x4.x * a_r - x4.y * a_th) / r1;
			//a_y += (x4.y * a_r + x4.x * a_th) / r1;
			
			//******************************************
			//***************************
			//saturation factor 
			p = 2.0/3.0*pow(m_pl/Msun,0.75)/pow(2.0*PI*alpha_t,0.5)*pow(hgas,1.75);
			double pp = p/pow(8.0/(45.0*PI),0.5);
			double ppp = p/pow(28.0/(45.0*PI),0.5);
			double fp = 1.0/(1.0+pow(p/1.3,2));
			double pps = 1.0/(1.0+pow(pp,4));
			double gpp = pps*(16.0*pow(pp,1.5)/25.0)+(1.0-pps)*(1.0-9.0*pow(pp,-2.67)/25.0);
			double ppps = 1.0/(1.0+pow(ppp,4));
			double akppp = ppps*(16.0*pow(ppp,1.5)/25.0)+(1.0-ppps)*(1.0-9.0*pow(ppp,-2.67)/25.0);
			//!ftot
  		 	//factors of each torques in viscous region: -0.375, -1.125       
			double flb_vis = -4.375;
			double fhb_vis = 1.2375;
			double fhe_vis = 5.5018;
			double fcb_vis = 0.7875;
			double fce_vis = 1.17;
			//factors of each torques in irr region: -3/7, -15/14             
			double flb_irr = -3.08;
			double fhb_irr = 0.47;
			double fhe_irr = 0;
			double fcb_irr = 0.3;
			double fce_irr = 0.0;
			//******************
			//  if (opt_corotation_damping == 'True')   then 
			//consider the corotation torque attenuation due to high eccentricity and inclination Bitsch&Kley2010,Fendyke&Nelson2014 
			double f1;
			double f_ecc = 1; //exp(-ecc)  need to calculate ecc and inc!!! future 
			double f_inc = 1; //
			double f1_vis,f1_irr;
			if (1) {
				f1_vis = flb_vis+(fhb_vis+fhe_vis*fp)*fp*gpp+(1-akppp)*(fcb_vis+fce_vis)*f_ecc;
				f1_irr = flb_irr+(fhb_irr+fhe_irr*fp)*fp*gpp+(1-akppp)*(fcb_irr+fce_irr)*f_ecc;
	 		}else{
				f1_vis = flb_vis + (fhb_vis + fhe_vis*fp)*fp*gpp + (1 - akppp)*(fcb_vis + fce_vis);
				f1_irr = flb_irr + (fhb_irr + fhe_irr*fp)*fp*gpp + (1 - akppp)*(fcb_irr + fce_irr); 
			}     
			//rtrans = 60;//transition radius between two regions  
			fs = 1/(1+pow(r1/rtrans,4));
			if (1){ // include viscous heating region
				f1 = (f1_vis* siggas_vis*fs + f1_irr*siggas_irr*(1-fs))/siggas ;
			}else{
				f1 = f1_irr;
			}
			//printf("\n f1:%g,%g\n",f1_irr,f1_vis);
			//ensure it is not zero
			if (abs(f1)<1e-10){
				f1 = 1e-10;
			}
			double f2;
			if(1){
				f2=-1e-10;//不考虑type 2
			}else{
				f2=-1.0;
			}
			//******************
			//TYPE I migration
			//*********************
			//double v_kep = sqrt(Msun * def_ksq / r1 - a_r * r1); 
			//Omega[MSUN^(1/2)/AU^(3/2)]
			//sigma单位本就是AU MSUN
			//printf("sigma?%g\n",Sigma);
			//Sigma=Sigma/((100*def_AU)*(100*def_AU)/(1000*def_Solarmass));
			Sigma = siggas;
			//printf("sigma?%g,%g,%g,%g,%g\n",r1,siggas,Sigma,siggas_vis,siggas_irr);
			//printf("depfac:%g",depfac);
			//T=2.8e2*pow(r1,-0.5)
			//cs=1.0e5*pow(r1,-0.25)[cm/s]  *(3600*24*365.25s/yrs)*(1AU/def_AU*100 cm)
			double cs = 1.0e5*pow(r1,-0.25)*(3600*24*365.25)/(def_AU*100);
			double tauwave = (Msun/m)*(Msun/(Sigma*pow(r1,2)))*pow(cs/r1/Omega,4)/Omega;
			//double tauwave = (Msun/m)*(Msun/(Sigma*pow(r1,2)))*pow(hgas,4)/Omega;//this sigma is gassigma,这里单位是年吗
			//printf("fact:%g,%g,%g,%g\n",m,pow(r1,2),hgas,Omega);
			//printf("sigma:%g,%g\n",Sigma,siggas);
			//printf("h:%g,%g\n",h,hgas);
			double f_tot=1.0e-20;//if no migration it is a small number
			f_tot = f1;//only type I
			f_tot  = f1*fs+f2*(1.0-fs)/pow(m_pl/mgo,2);//确认一下后面这个的数量级，应该没问题，但是极大的影响数量级
			//printf("f_tt:%g,%g,%g\n",fs,f1*fs,f2*(1.0-fs)/pow(m_pl/mgo,2));
			//printf("%g,%g",m_pl,mgo);

			//migration timescale f_tot>0, outward migration; f_tot<0, inward migration 
			double taumig = 0.5*tauwave/f_tot/pow(hgas,2);//
			double tauecc = tauwave/0.78/abs(f_tot);
			double tauinc = tauwave/0.544/abs(f_tot);//有更复杂的式子(cresswell&nelson2009)
			double rv = x4.x*v4.x+x4.y*v4.y+x4.z*v4.z;
			//printf("tau:wave,mig,ecc,inc%g,%g,%g,%g\n",tauwave,taumig,tauecc,tauinc);
			//
			//
			double3 asemi;
			double3 aecc;
			double3 acc;
			
			asemi.x=v4.x/taumig;
			asemi.y=v4.y/taumig;
			asemi.z=v4.z/taumig;
			
			aecc.x=-2*rv*x4.x/pow(r,2)/tauecc;
			aecc.y=-2*rv*x4.y/pow(r,2)/tauecc;
			aecc.z=-2*rv*x4.z/pow(r,2)/tauecc;
			
			acc.x = asemi.x+aecc.x;
			acc.y = asemi.y+aecc.y;
			acc.z = asemi.z+aecc.z;
			
			double ainc = -v4.z/tauinc;
			acc.z = acc.z+ainc;
			//printf("migration timescale:%g\n",sqrt(pow(x4.x,2)+pow(x4.y,2))/sqrt(pow(v4.x,2)+pow(v4.y,2)));
			//a_x += acc.x;
			//a_y += acc.y;
			//a_z += acc.z;
			
			//double times_mig=m*Omega*r1*r1/2/taumig;
			
			//*************
			//turbulence disk
			//***************
			//!t      = current epoch [days]
			//!mstar  = star mass (in solar masses * K2)
			//!num    = current number of bodies
			//!mpl    = planet mass (in solar masses * K2)
			//!x      = coordinates (x,y,z) with respect to the central body [AU]
			//!v      = velocities (vx,vy,vz) with respect to the central body [AU/day]
			//!t0     = turblent mode's initial time  [day]
			//!delta_t= turblent mode's duration time  [day]
			//!r_c    = turblent mode's radial distance  [AU]
			//!theta_c= turblent mode's azimutal angle 
			//!wave   = turblent mode's wave number  
			//!ksi   = turblent mode's coefficient  
			//!Omega_c   = turblent mode's angular frequence [day^-1]  
			//!rextend_c   = turblent mode's raidial extent [AU] 
			
			double wave[50];
			double t_0[50];
			double delta_t[50];
			double r_c[50];
			double theta_c[50];
			double ksi[50];
			double Omega_c[50];
			double rextend_c[50];
			double c_s;
			double rand1,rand2,rand3,rand4,rand5;
			curandState state;
			curand_init((unsigned long long)clock()+id, 0, 0, &state);
			rand1=curand_uniform_double(&state);
			rand2=curand_uniform_double(&state);
			rand3=curand_uniform_double(&state);
			rand4=curand_uniform_double(&state);
			rand5=curand_uniform_double(&state);
			//printf("%g,%g,%g,%g,%g\n",rand1,rand2,rand3,rand4,rand5);
			
			//dTime
			//if(dTime == 0.0){
			//	;//这里是用来赋值为零，但c默认设零所以无所谓
			//}
			for(int i=1;i<=50;i++){
				//printf("time?%g,%g\n",dTime*365.25,t_0[i]+delta_t[i]);
				if(1){
				//if(dTime*365.25>=t_0[i]+delta_t[i]){//理解一下为什么这么判断
					ksi[i]=sqrt(-2.0*log(rand1))*cos(2.0*PI*rand2)/sqrt(2*PI);//原文是TWOPI，应该是这个吧
					r_c[i]=exp(rand3*(log(rg1)-log(1.1*rg0))+log(1.1*rg0));//回头确认一下ro rin对应什么
					theta_c[i]=rand4*2*PI;
					wave[i]=1.0*int(pow(e,rand5*(log(64.0)-log(2.0))+log(2.0)));
					Omega_c[i]=pow((Msun/pow(r_c[i],3)),0.5);
					c_s=hgas*r_c[i]*Omega_c[i];
					delta_t[i]=2*PI*r_c[i]/wave[i]/c_s;
					rextend_c[i]=PI*r_c[i]/4.0/wave[i];
					//printf("%g\n",c_s);
				}
			}
			//printf("%g,%g\n",Sigma,siggas);
			//printf("fact%g,%g,%g,%g,%g\n",ksi[1],r_c[1],theta_c[1],wave[1],Omega_c[1]);
			//****************************************
			//turbulence force
			//************************
			double lambda,lambda_cm,lambda_sm;
			double theta = acos(x4.x/r1);
			double sigma_rmode=0,sigma_tmode=0;
			double f_gamma=64.0*Sigma*pow(r1,2)/pow(PI,2)/Msun;//这里的sigma对应siggas
			double f_ts=8.5e-2*hgas*sqrt(alpha_vis);//f_ts 对应文章的小gamma，反映strength of the turbulence
			double f_acc=f_ts*f_gamma*r1*pow(Omega,2);
			//printf("fac:,%g,%g,%g\n",f_gamma,f_ts,f_acc);
			for(int i=1;i<=50;i++){
			//printf("%g",wave[i]);
				if(wave[i]>=6.0){
					lambda_cm=0.0;
					lambda_sm=0.0;
				}else{
					lambda=ksi[i]*pow(e,-pow((r1-r_c[i]),2)/pow(rextend_c[i],2))*sin(PI*(dt_d[st] -t_0[i])/delta_t[i]);
					lambda_cm=lambda*cos(wave[i]*theta-theta_c[i]-Omega_c[i]*(dt_d[st] -t_0[i]));
					lambda_sm=lambda*sin(wave[i]*theta-theta_c[i]-Omega_c[i]*(dt_d[st] -t_0[i]));
					//printf("%g\n",wave[i]*theta-theta_c[i]-Omega_c[i]*(dt_d[st] -t_0[i]));
					//printf("%g,%g \t",lambda_cm,lambda_sm);
				}
				sigma_rmode=sigma_rmode+(1.0+2.0*r1*(r1-r_c[i])/pow(rextend_c[i],2))*lambda_cm;
				sigma_tmode=sigma_tmode+wave[i]*lambda_sm;
				//printf("%g,%g \t",lambda_cm,lambda_sm);
				//printf("%g,%g \t",1.0+2.0*r1*(r1-r_c[i]),pow(rextend_c[i],2));
			}
			//printf("lambda:%g,%g,%g\n",lambda,lambda_cm,lambda_sm);
			if(isnan(sigma_rmode)){//这里的nan真的很诡异,居然受printf影响
				sigma_rmode=0;//printf("\n !nan!\n \n \n \n \n \n \n \n \n \n !nan!\n \n \n \n \n \n \n \n \n");
			}
			if(isnan(sigma_tmode)){
				sigma_tmode=0;//printf("\n !nan!\n \n \n \n \n \n \n \n \n \n !nan!\n \n \n \n \n \n \n \n \n");
			}
			double acctur_r=f_acc*sigma_rmode;
			double acctur_t=f_acc*sigma_tmode;
			double3 acctur;
			acctur.x=acctur_r*cos(theta)-acctur_t*sin(theta);
			acctur.y=acctur_r*sin(theta)+acctur_t*cos(theta);
			acctur.z=0.0;
			
			//printf("\n mode:%g,%g,%g\n",sigma_rmode,sigma_tmode,f_acc);
			//printf("r,t,x,y,z:%g,%g,%g,%g,%g\n",acctur_r,acctur_t,acctur.x,acctur.y,acctur.z);
			//a_x += acctur.x;
			//a_y += acctur.y;
			//a_z += acctur.z;
			
		}//end if ir
		
		if(x4.w >= 0.0){
			double v2 = v4.x * v4.x + v4.y * v4.y + v4.z * v4.z;
			//Kick
			v4.x += a_x * dt;
			v4.y += a_y * dt;
			v4.z += a_z * dt;
//printf("Gas 2 %d %g %g %g %g %g %g %g\n", id, v4.x, v4.y, v4.z, a_x, a_y, a_z, x4.w);
			double v2B = v4.x * v4.x + v4.y * v4.y + v4.z * v4.z;

			v4_d[id] = v4;
	
			if(x4.w > 0.0){
				U = 0.5 * x4.w * (v2 - v2B);
				EnergySum_d[id] += U;
			}

		}
	}
}

// *************************************************
// This kernel uses a file to read the gas disk structure
//
// ****************************************************
__global__ void GasAcc2_kernel(double4 *x4_d, double4 *v4_d, int *index_d, double *time_d, double2 *Msun_d, double *dt_d, const int N, double *EnergySum_d, const int Nst, const double Ct, const int nr, double2 GasDatatime, double4 *GasData_d, int UsegasPotential, int UsegasEnhance, int UsegasDrag, int UsegasTidalDamping, const double Mgiant, const int Nstart){

	int id = blockIdx.x * blockDim.x + threadIdx.x + Nstart;

	double rin = 0.25;
	double rout = 1000.0;

	double G_alpha =     Gas_parameters_c[1];
	//double G_beta =      Gas_parameters_c[2];

	if(id < N + Nstart){

		int st = 0;
		if(Nst > 1 && id < N + Nstart) st = index_d[id] / def_MaxIndex;

		double rs = log(rout / rin) / ((double)(nr - 1)); //slope in distance profile 

		double dt = dt_d[st] * Ct;
		double dTime = time_d[st] / 365.25;

//if(id == 0) printf("rs %g %g %g %g\n", rs, GasDatatime.x, GasDatatime.y, dTime);

		double Msun = Msun_d[st].x;
		double U = 0.0;

		double3 v_rel3;
		double v_rel_r, v_rel_th;

		double4 x4 = x4_d[id];
		double4 v4 = v4_d[id];

		double r1 = x4.x * x4.x + x4.y * x4.y;
		double rsq = r1 + x4.z * x4.z;
		double vsq = v4.x * v4.x + v4.y * v4.y + v4.z * v4.z;
		r1 = sqrt(r1);
		double r = sqrt(rsq);

		volatile double a_r = 0.0;
		volatile double a_th = 0.0;
		volatile double a_x = 0.0;
		volatile double a_y = 0.0;
		volatile double a_z = 0.0;

	
		if(r1 > rin && r1 < rout){ //otherwise there is no gas

			int ri = (int)(log(r1 / rin) / rs);
			double ri0 = rin * exp(rs * ri);
			double ri1 = rin * exp(rs * (ri + 1));

			double4 GasData0 = GasData_d[ri];
			double4 GasData1 = GasData_d[ri + 1];

			double tr = (r1 - ri0) / (ri1 - ri0);
			double Sigma0 = (GasData1.x - GasData0.x) * tr + GasData0.x;
			double h0 = (GasData1.y - GasData0.y) * tr + GasData0.y;
			double Sigma1 = (GasData1.z - GasData0.z) * tr + GasData0.z;
			double h1 = (GasData1.w - GasData0.w) * tr + GasData0.w;
			double tt = (dTime - GasDatatime.x) / (GasDatatime.y - GasDatatime.x);
			double Sigma = (Sigma1 - Sigma0) * tt + Sigma0;
			Sigma *= 1.49598*1.49598/1.98892*1.0e-7;
			double h = ((h1 - h0) * tt + h0) * r1;

//if(id < 100) printf("%d %g %g %g %g %g\n", id, r1, Sigma, h, GasData1.x, GasData0.x);

			double zh = x4.z / h;

			double m = x4.w;
			if(m == 0.0) m = def_MgasSmall;
			if(Sigma > 0.0){
			
				//Enhanced Drag
				double Soft;
				if(UsegasEnhance == 1 || (UsegasEnhance == 2 && m < Mgiant)){
					
					if(m < def_M_Enhance && m > def_MgasSmall){
						double pid = log(m/def_fMass_min) / log(def_M_Enhance/def_fMass_min);
						double jc = def_M_Enhance/def_Mass_pl;
						m = pow(jc, pid);
						m *= def_Mass_pl;
						Soft = v4.w * pow(m/x4.w, 1.0/3.0);
					}
					else{
					
						Soft = v4.w;
					}
				}
				else{
					Soft = v4.w;

				}

				//vKep
				double v_kep = 0.0;
				if(UsegasDrag > 0 || UsegasTidalDamping > 0){
					v_kep = sqrt(Msun * def_ksq / r1 - a_r * r1); 
				}

				//gas drag
				if(UsegasDrag == 1 || (UsegasDrag == 2 && m < Mgiant)){
					double eta = 0.5 * ((G_alpha + 1.75) * h * h + 0.5 * x4.z * x4.z) / (r1 * r1);
					double v_gas = v_kep * (1.0 - eta);	//Change that to v_kep * sqrt(1.0 - 2.0 * eta);

					double rho = facrho * Sigma / h * exp(-0.5 * zh * zh);
					v_rel3.x = v4.x + v_gas * x4.y / r1; 
					v_rel3.y = v4.y - v_gas * x4.x / r1;
					v_rel3.z = v4.z;

					v_rel_r = (x4.x * v_rel3.x + x4.y * v_rel3.y) / r1;
					v_rel_th = (x4.x * v_rel3.y - x4.y * v_rel3.x) / r1;
					
					double v_rel = v_rel3.x * v_rel3.x + v_rel3.y * v_rel3.y + v_rel3.z * v_rel3.z;
					v_rel = sqrt(v_rel);
					if(m > 0.0) v_rel *= M_PI / (2.0 * m) * def_Gas_cd * Soft * Soft * rho;
					else v_rel = 0.0;

					a_x += -v_rel * v_rel3.x;
					a_y += -v_rel * v_rel3.y;
					a_z += -v_rel * v_rel3.z;
				}

				//tidal damping
				if(UsegasTidalDamping == 1 || (UsegasTidalDamping == 2 && m < Mgiant)){
					v_rel3.x = v4.x + v_kep * x4.y / r1;
					v_rel3.y = v4.y - v_kep * x4.x / r1;
					v_rel3.z = v4.z;

					v_rel_r = (x4.x * v_rel3.x + x4.y * v_rel3.y) / r1;
					v_rel_th = (x4.x * v_rel3.y - x4.y * v_rel3.x) / r1;

					double Mtot = Msun + m;
					double Etot = 0.5 * vsq - def_ksq * Mtot / r;
					double3 L;
					L.x = x4.y * v4.z - x4.z * v4.y;
					L.y = x4.z * v4.x - x4.x * v4.z;
					L.z = x4.x * v4.y - x4.y * v4.x;

					double Lsq = L.x * L.x + L.y * L.y + L.z * L.z;
					double esq = 1.0 + 2.0 * Etot * Lsq / (Mtot * Mtot);
					if(esq < 0.0) esq = 0.0;
					double isq = 1.0 - L.z * L.z / Lsq;
					double a = -0.5 / Etot;

					double r1h2 = r1 * r1 / (h * h);
					
					double chi = 0.5 * a * sqrt(esq + isq) / h;
					double chi3 = chi * chi * chi;

					double iTau_tid1 = m * (Sigma * r1 * r1) * r1h2	/ (r1 * sqrt(r1)); //a = ri because of circular orbit
					double iTauwave_gwave = iTau_tid1 * r1h2 / (1.0 + 0.25 * chi3);
					iTau_tid1 *= 3.8 * (1.0 - 0.683 * chi3 * chi) / (1.0 + 0.269329 * chi3 * chi * chi);

					a_th += -0.5 * iTau_tid1 * v_kep; //circular

					a_r += (0.104 * v_rel_th + 0.176 * v_rel_r) * iTauwave_gwave;
					a_th += (-1.736 * v_rel_th + 0.325 * v_rel_r) * iTauwave_gwave;
					a_z += (-1.088 * v_rel3.z - 0.871 * v_kep * x4.z / r1) * iTauwave_gwave;
				}
			}
			a_x += (x4.x * a_r - x4.y * a_th) / r1;
			a_y += (x4.y * a_r + x4.x * a_th) / r1;
		}

		if(x4.w >= 0.0){

			double v2 = v4.x * v4.x + v4.y * v4.y + v4.z * v4.z;
			//Kick
			v4.x += a_x * dt;
			v4.y += a_y * dt;
			v4.z += a_z * dt;

			double v2B = v4.x * v4.x + v4.y * v4.y + v4.z * v4.z;

			v4_d[id] = v4;
			if(x4.w > 0.0){
				U = 0.5 * x4.w * (v2 - v2B);
				EnergySum_d[id] += U;
			}
		}
	}
}

// ************************************
// this kernel sums up all the Energy loss due to the Gas Disc and adds to the internal Energy
//
// It works for the case of multiple blocks
// Must be followed by gasEnergyd2
//
// using vold as temporary storage
//
// Author: Simon Grimm
// March 2021
// *************************************
__global__ void gasEnergyd1_kernel(double *EnergySum_d, double4 *vold_d, const int N){

	int id = blockIdx.x * blockDim.x + threadIdx.x;

	double U = 0.0;

	extern __shared__ double gasd1_s[];
	double *U_s = gasd1_s;

	int lane = threadIdx.x % warpSize;
	int warp = threadIdx.x / warpSize;

	if(warp == 0){
		U_s[threadIdx.x] = 0.0;
	}

	for(int i = 0; i < N; i += blockDim.x * gridDim.x){     //gridDim.x is for multiple block reduce
		if(id + i < N){
			U += EnergySum_d[id + i];
			EnergySum_d[id + i] = 0.0;
		}
	}
	__syncthreads();

	for(int i = 1; i < warpSize; i*=2){
#if def_OldShuffle == 0
		U += __shfl_xor_sync(0xffffffff, U, i, warpSize);
#else
		U += __shfld_xor(U, i);
#endif
	}

	__syncthreads();

	if(blockDim.x > warpSize){
		//reduce across warps
		if(lane == 0){
			U_s[warp] = U;
		}

		__syncthreads();
		//reduce previous warp results in the first warp
		if(warp == 0){
			U = U_s[threadIdx.x];
			for(int i = 1; i < warpSize; i*=2){
#if def_OldShuffle == 0
				U += __shfl_xor_sync(0xffffffff, U, i, warpSize);
#else
				U += __shfld_xor(U, i);
#endif
			}
		}

	}
	__syncthreads();
	if(threadIdx.x == 0){
		vold_d[blockIdx.x].x = U;
//printf("Ud1 %d %.20g\n", blockIdx.x, U);
	}

}

// *********************************************************
// This kernel reads the result from the multiple thread block kernel gasEnergyd1_kernel
// and performs the last summation step in
// --a single thread block --
//
// using vold as temporary storage
//
// Author: Simon Grimm
// March 2021
// *************************************
__global__ void gasEnergyd2_kernel(double *U_d, double4 *vold_d, const int N){

	int idy = threadIdx.x;

	double U = 0.0;

	extern __shared__ double gasd2_s[];
	double *U_s = gasd2_s;

	int lane = threadIdx.x % warpSize;
	int warp = threadIdx.x / warpSize;

	if(warp == 0){
		U_s[threadIdx.x] = 0.0;
	}

	if(idy < N){
		U += vold_d[idy].x;
	}
//printf("u %d %g\n", idy, U);
	__syncthreads();

	for(int i = 1; i < warpSize; i*=2){
#if def_OldShuffle == 0
		U += __shfl_xor_sync(0xffffffff, U, i, warpSize);
#else
		U += __shfld_xor(U, i);
#endif
	}

	__syncthreads();

	if(blockDim.x > warpSize){
		//reduce across warps
		if(lane == 0){
			U_s[warp] = U;
		}

		__syncthreads();
		//reduce previous warp results in the first warp
		if(warp == 0){
			U = U_s[threadIdx.x];
			for(int i = 1; i < warpSize; i*=2){
#if def_OldShuffle == 0
				U += __shfl_xor_sync(0xffffffff, U, i, warpSize);
#else
				U += __shfld_xor(U, i);
#endif
			}
		}

	}
	__syncthreads();
	if(threadIdx.x == 0){
		U_d[0] += U;
//printf("Ud2 %.20g %.20g\n", U, U_d[0]);
	}

}


// ************************************
// this kernel sums up all the Energy loss due to the Gas Disc and adds to the internal Energy
//
// It works for the case of multiple warps, but only 1 thread block
//
// Author: Simon Grimm
// March 2021
// *************************************
__global__ void gasEnergya_kernel(double *EnergySum_d, double *U_d, const int N){

	int idy = threadIdx.x;

	double U = 0.0;

	for(int i = 0; i < N; i += blockDim.x * gridDim.x){     //gridDim.x is for multiple block reduce
		if(idy + i < N){
			U += EnergySum_d[idy + i];
			EnergySum_d[idy + i] = 0.0;
		}
	}
//printf("%d %g\n", idy, U_s[idy]);
	__syncthreads();

	for(int i = 1; i < warpSize; i*=2){
#if def_OldShuffle == 0
		U += __shfl_xor_sync(0xffffffff, U, i, warpSize);
#else
		U += __shfld_xor(U, i);
#endif
	}

	__syncthreads();

	if(blockDim.x > warpSize){
		//reduce across warps
		extern __shared__ double gasa_s[];
		double *U_s = gasa_s;

		int lane = threadIdx.x % warpSize;
		int warp = threadIdx.x / warpSize;
		if(warp == 0){
			U_s[threadIdx.x] = 0.0;
		}
		__syncthreads();

		if(lane == 0){
			U_s[warp] = U;
		}

		__syncthreads();
		//reduce previous warp results in the first warp
		if(warp == 0){
			U = U_s[threadIdx.x];
			for(int i = 1; i < warpSize; i*=2){
#if def_OldShuffle == 0
				U += __shfl_xor_sync(0xffffffff, U, i, warpSize);
#else
				U += __shfld_xor(U, i);
#endif
			}
			if(lane == 0){
				U_s[0] = U;
			}
		}
		__syncthreads();

		U = U_s[0];
	}
	__syncthreads();


	if(idy == 0){
		U_d[0] += U;
//printf("Ua %.20g %.20g\n", U, U_d[0]);
	}
}


// ************************************
// this kernel sums up all the Energy loss due to the Gas Disc and adds to the internal Energy
//
//It works for the case of only 1 single warp
//
// Author: Simon Grimm
// March 2021
// *************************************
__global__ void gasEnergyc_kernel(double *EnergySum_d, double *U_d, const int N){

	int idy = threadIdx.x;

	double U = 0.0;

	if(idy < N){
		U = EnergySum_d[idy];
		EnergySum_d[idy] = 0.0;
	}
//printf("%d %g\n", idy, U_s[idy]);
	__syncthreads();

	for(int i = 1; i < warpSize; i*=2){
#if def_OldShuffle == 0
		U += __shfl_xor_sync(0xffffffff, U, i, warpSize);
#else
		U += __shfld_xor(U, i);
#endif
	}

	__syncthreads();

	if(idy == 0){
		U_d[0] += U;
//printf("Uc %.20g %.20g\n", U, U_d[0]);
	}
}

#if def_CPU == 1
void gasEnergy_cpu(double *EnergySum_h, double *U_h, const int N){

	double U = 0.0;

	for(int id = 0; id < N; ++id){
		U += EnergySum_h[id];
		EnergySum_h[id] = 0.0;
	}
//printf("%d %g\n", idy, U_s[idy]);
	U_h[0] += U;
//printf("Uc %.20g %.20g\n", U, U_d[0]);
}
#endif


// ************************************
// This kernel sums up all the Energy loss due to the Gas Disc and adds to the internal Energy
// Uses paralel reduction sum with warp shuffle operations
// Author: Simon Grimm
// February 2023
// *************************************
__global__ void gasEnergy_kernel(double *EnergySum_d, double *U_d, const int N){

	int idy = threadIdx.x;

	double U = 0.0;

	for(int i = 0; i < N; i += blockDim.x){
		if(idy + i < N){
			U += EnergySum_d[idy + i];
			EnergySum_d[idy + i] = 0.0;
		}
	}
//printf("%d %g\n", idy, U_s[idy]);
	__syncthreads();

	for(int i = 1; i < warpSize; i*=2){
#if def_OldShuffle == 0
		U += __shfl_xor_sync(0xffffffff, U, i, warpSize);
#else
		U += __shfld_xor(U, i);
#endif
	}
	__syncthreads();

	if(blockDim.x > warpSize){
		//reduce across warps
		extern __shared__ double UEM_s[];
		double *U_s = UEM_s;

		int lane = threadIdx.x % warpSize;
		int warp = threadIdx.x / warpSize;
		if(warp == 0){
			U_s[threadIdx.x] = 0.0;
		}
		__syncthreads();

		if(lane == 0){
			U_s[warp] = U;
		}

		__syncthreads();
		//reduce previous warp results in the first warp
		if(warp == 0){
			U = U_s[threadIdx.x];
			for(int i = 1; i < warpSize; i*=2){
#if def_OldShuffle == 0
				U += __shfl_xor_sync(0xffffffff, U, i, warpSize);
#else
				U += __shfld_xor(U, i);
#endif
			}
			if(lane == 0){
				U_s[0] = U;
			}
		}
		__syncthreads();

		U = U_s[0];
	}

	__syncthreads();


	if(idy == 0){
		U_d[0] += U;
//printf("U %.20g %.20g\n", U_s[0], U_d[0]);
	}
}

//This function calls the Gas Energy kernel
__host__ void Data::gasEnergyCall(){
	int NN = N_h[0] + Nsmall_h[0];
#if def_CPU == 0
	if(NN <= WarpSize){
		gasEnergyc_kernel <<< 1, WarpSize, 0, hstream[0] >>> (EnergySum_d, U_d, NN);
	}
	else if(NN <= 512){
		int nn = (NN + WarpSize - 1) / WarpSize;
		gasEnergya_kernel <<< 1, nn * WarpSize, WarpSize * sizeof(double), hstream[0] >>> (EnergySum_d, U_d, NN);
	}
	else{
		int nct = 512;
		int ncb = min((NN + nct - 1) / nct, 1024);
		gasEnergyd1_kernel <<< ncb, nct, WarpSize * sizeof(double), hstream[0] >>> (EnergySum_d, vold_d, NN);
		gasEnergyd2_kernel <<< 1, ((ncb + WarpSize - 1) / WarpSize) * WarpSize, WarpSize * sizeof(double), hstream[0] >>> (U_d, vold_d, ncb);
	}
#else
	gasEnergy_cpu (EnergySum_h, U_h, NN);

#endif
}
__host__ void Data::gasEnergyMCall(int st){
	int NBS = NBS_h[st];
	gasEnergy_kernel <<< 1, NB[st], WarpSize * sizeof(double), hstream[st%16] >>> (EnergySum_d + NBS, U_d + st, N_h[st]);
}

__host__ void Data::GasAccCall(double *time_d, double *dt_d, double Ct){
	int nt = min(32, NB[0]);
	GasAcc_kernel <<< (N_h[0] + nt - 1) / nt , nt >>> (x4_d, v4_d, index_d, GasDisk_d, GasAcc_d, time_d, Msun_d, dt_d, N_h[0], EnergySum_d, Nst, Ct, P.UsegasPotential, P.UsegasEnhance, P.UsegasDrag, P.UsegasTidalDamping, P.G_Mgiant, 0, G_Nr_p,0);
}
__host__ void Data::GasAccCall_small(double *time_d, double *dt_d, double Ct){
	if(Nsmall_h[0] > 0) GasAcc_kernel <<<(Nsmall_h[0] + 127)/128, 128 >>> (x4_d + N_h[0], v4_d + N_h[0], index_d + N_h[0], GasDisk_d, GasAcc_d, time_d, Msun_d, dt_d, Nsmall_h[0], EnergySum_d, Nst, Ct, P.UsegasPotential, P.UsegasEnhance, P.UsegasDrag, P.UsegasTidalDamping, P.G_Mgiant, 0, G_Nr_p,0);
}
__host__ void Data::GasAccCall_M(double *time_d, double *dt_d, double Ct){
	GasAcc_kernel <<< (NT + 127) / 128, 128 >>> (x4_d, v4_d, index_d, GasDisk_d, GasAcc_d, time_d, Msun_d, dt_d, NT, EnergySum_d, Nst, Ct, P.UsegasPotential, P.UsegasEnhance, P.UsegasDrag, P.UsegasTidalDamping, P.G_Mgiant, Nstart, G_Nr_p,0);
}
__host__ void Data::GasAccCall2_small(double *time_d, double *dt_d, double Ct){
	if(Nsmall_h[0] > 0) GasAcc2_kernel <<<(Nsmall_h[0] + 127)/128, 128 >>> (x4_d + N_h[0], v4_d + N_h[0], index_d + N_h[0], time_d, Msun_d, dt_d, Nsmall_h[0], EnergySum_d, Nst, Ct, GasDatanr, GasDatatime, GasData_d, P.UsegasPotential, P.UsegasEnhance, P.UsegasDrag, P.UsegasTidalDamping, P.G_Mgiant, 0);
}

__host__ int Data::freeGas(){
	cudaError_t error;

	cudaFree(Gas_rg_d);
	cudaFree(Gas_zg_d);
	cudaFree(Gas_rho_d);
	cudaFree(GasDisk_d);
	cudaFree(GasAcc_d);

#if def_CPU ==1
	free(Gas_rg_h);
	free(Gas_zg_h);
	free(Gas_rho_h);
	free(GasDisk_h);
	free(GasAcc_h);

#endif
	error = cudaGetLastError();
	if(error != 0){
		printf("Cuda Gas free error = %d = %s\n",error, cudaGetErrorString(error));
		fprintf(masterfile, "Cuda Gas free error = %d = %s\n",error, cudaGetErrorString(error));
		return 0;
	}
	return 1;

}

