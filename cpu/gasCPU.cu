#include "Orbit2CPU.h"
#define facrho 1.0/sqrt(2.0 * M_PI)


double *Gas_rg_h;
double *Gas_zg_h;
double *Gas_rho_h;
double3 *GasDisk_h;
double3 *GasAcc_h;


double Gas_parameters_h[8];
double Gas_parameters_c[8];

int G_Nr_g, G_Nr_p;

void Data::GasAlloc(){

	G_Nr_g = (P.G_rg1 - P.G_rg0 + 0.5 * P.G_drg) / P.G_drg - 1;
	G_Nr_p = (P.G_rp1 - P.G_rg0 + 0.5 * P.G_drg) / P.G_drg + 1;

	printf("Gas N %d %d\n", G_Nr_g, G_Nr_p);


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

#if def_CPU == 0
	cudaMemcpyToSymbol(Gas_parameters_c, Gas_parameters_h, 8 * sizeof(double), 0, cudaMemcpyHostToDevice);
#else
	memcpy(Gas_parameters_c, Gas_parameters_h, 8 * sizeof(double));
#endif
}


// *************************************************
// This function corresponds to the msrGasTable function in the file master.c in pkdgrav_planets.
//
// ****************************************************
void GasDisk_cpu(double *Gas_rg_h, double *Gas_zg_h, double *Gas_rho_h, const int G_Nr_g){

	int ig = 0 * 1 + 0; // r
	int jg = 0 * 1 + 0; // z

	double G_alpha = Gas_parameters_c[1];
	double G_beta =  Gas_parameters_c[2];
	double G_Sigma10 = Gas_parameters_c[3];
	double rg0 = Gas_parameters_c[5];	//inner edge of the gas disk
	double drg = Gas_parameters_c[7];	//spacing of the gas disk

//	double rin, ro;
//	if(uniform != 1){
//		rin = 0.1 + dTime/(2.0 * M_PI * dTau_diss); // time scale for the inner edge to move 1AU
//	}

	for(ig = 0 * 1 + 0; ig < G_Nr_g; ++ig){
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
		for(jg = 0 * 1 + 0; jg < def_Gasnz_g; ++jg){
			if(jg == 0){
				Gas_rg_h[ig] = rg;
//printf("Gas_rg %d %g\n", ig, Gas_rg_h[ig]);
			}

			double zg = 0.01 * (0.5 + jg) * rg;
			double zh = zg / h;
			Gas_zg_h[ig * def_Gasnz_g + jg] = zg;
			Gas_rho_h[ig * def_Gasnz_g + jg] = facrho * (Sigma / h) * exp(-0.5 * zh * zh);
//printf("Gas_rho %g %g %g\n", rg, zg, Gas_rho_h[ig * def_Gasnz_g + jg]);
		}
	}
}

// **********************************************
// First Kind elliptic integral
// This function corresponds to the rf function in the file master.c in pkdgrav_planets.
// It is based on numerical recipes and the paper from Carlson 
// **********************************************
double rf(double x, double y, double z){
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
double rd(double x, double y, double z){
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
void gasTable_cpu(double *Gas_rg_h, double *Gas_zg_h, double *Gas_rho_h, double3 *GasDisk_h, double3 *GasAcc_h, const int G_Nr_g, const int G_Nr_p){

	int ip = 0 * 1 + 0; // r
	int jp = 0 * 1 + 0; // z

	volatile double ar, az;
	double rp, zp;
	double ellf, elle;

	double G_alpha = Gas_parameters_c[1];
	double G_beta =  Gas_parameters_c[2];
	double G_Sigma10 = Gas_parameters_c[3];
	double rg0 = Gas_parameters_c[5];	//inner edge of the gas disk
	double drg = Gas_parameters_c[7];	//spacing of the gas disk

	for(ip = 0 * 1 + 0; ip < G_Nr_p; ++ip){	
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
			GasDisk_h[ip].x = Sigma;
			GasDisk_h[ip].y = h;
			GasDisk_h[ip].z = rp;
//printf("GasDisk_h %d %g %g %g\n", ip, rp, h, Sigma);
		}
	}

	for(ip = 0 * 1 + 0; ip < G_Nr_p; ++ip){
		rp = rg0 + drg * ip;
		for(jp = 0 * 1 + 0; jp < def_Gasnz_p; ++jp){

			zp = (0.03 * jp) * rp;
			ar = 0.0;
			az = 0.0;
			for(int ig = 0; ig < G_Nr_g; ++ig){
				double rgas = Gas_rg_h[ig];
				double dzg = 0.03 * rgas;
				
				for(int jg = 0; jg < def_Gasnz_g; ++jg){

					double zgas = Gas_zg_h[ig * def_Gasnz_g + jg];
					double rho_gas = Gas_rho_h[ig * def_Gasnz_g + jg];

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
			GasAcc_h[ip * def_Gasnz_p + jp].x = ar;
			GasAcc_h[ip * def_Gasnz_p + jp].y = az;
			GasAcc_h[ip * def_Gasnz_p + jp].z = zp;
//	printf("GasAcc %d %d %g %g %g\n", ip, jp, ar, az, zp);
		}
	}
}

int Data::setGasDisk(){
	int error;
	GasDisk_cpu /* NBGasDisk, NTGasDisk */ (Gas_rg_h, Gas_zg_h, Gas_rho_h, G_Nr_g);


	gasTable_cpu /* NBGasTabel, NTGasTabel */(Gas_rg_h, Gas_zg_h, Gas_rho_h, GasDisk_h, GasAcc_h, G_Nr_g, G_Nr_p);

	error = 0;
	fprintf(masterfile, "Gas Table error = %d = %s\n",error, "-");
	if(error != 0){
		printf("Gas Table error = %d = %s\n",error, "-");
		return 0;
	}
	else return 1;
}

// *************************************************
// This kernel corresponds to the pkdGasAccel function in the file pkd.c in pkdgrav_planets.
//
// ****************************************************
void GasAcc_cpu(double4 *x4_h, double4 *v4_h, int *index_h, double3 *GasDisk_h, double3 *GasAcc_h, double *time_h, double2 *Msun_h, double *dt_h, int N, double *EnergySum_h, int Nst, double Ct, int UsegasPotential, int UsegasEnhance, int UsegasDrag, int UsegasTidalDamping, double Mgiant, int Nstart, int G_Nr_p){

	int idy = 0;
	int id = 0 * 1 + idy + Nstart;

	double G_dTau_diss = Gas_parameters_c[0];
	double G_alpha =     Gas_parameters_c[1];
	double G_beta =      Gas_parameters_c[2];
	double G_Sigma10 =   Gas_parameters_c[3];
	double rg0 =         Gas_parameters_c[5];	//inner edge of the gas disk
	double rg1 =         Gas_parameters_c[6];	//outher edge of the gas disk
	double drg =         Gas_parameters_c[7];       //spacing of the gas disk

	for(id = 0 * 1 + idy + Nstart; id < N + Nstart; ++id){
		
		int st = 0;
		if(Nst > 1) st = index_h[id] / def_MaxIndex;
		double dt = dt_h[st] * Ct;
		double dTime = time_h[st] / 365.25;

		double Msun = Msun_h[st].x;
		double U = 0.0;

		double3 v_rel3;
		double v_rel_r, v_rel_th;
		double big = 1.0e7;

		double depfac = exp(-dTime/(G_dTau_diss));


		double4 x4 = x4_h[id];
		double4 v4 = v4_h[id];

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


	
		if(r1 > rg0 && r1 < rg1 && x4.z/r1 < 1.5){ //otherwise there is no gas

			double h = def_h_1 * r1 * pow(r1, G_beta);
			double Sigma = G_Sigma10 * pow(r1, -G_alpha);

			Sigma *= depfac;
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
					double rr0 = GasDisk_h[ip - 1].z;
					double rr1 = GasDisk_h[ip].z;

					int jp0 = floor(fabs(x4.z)/(rr0 * 0.03));
					int jp1 = floor(fabs(x4.z)/(rr1 * 0.03));

					double zz00 = GasAcc_h[(ip - 1) * def_Gasnz_p + jp0].z;
					double zz01 = GasAcc_h[(ip - 1) * def_Gasnz_p + jp0 + 1].z;
					double zz10 = GasAcc_h[ip * def_Gasnz_p + jp1].z;
					double zz11 = GasAcc_h[ip * def_Gasnz_p + jp1 + 1].z;

					double dr00 = ((x4.z - zz00)  * (x4.z - zz00)) + ((r1 - rr0)  * (r1 - rr0));
					double dr01 = ((x4.z - zz01)  * (x4.z - zz01)) + ((r1 - rr0)  * (r1 - rr0));
					double dr10 = ((x4.z - zz10)  * (x4.z - zz10)) + ((r1 - rr1)  * (r1 - rr1));
					double dr11 = ((x4.z - zz11)  * (x4.z - zz11)) + ((r1 - rr1)  * (r1 - rr1));
					
					dr00 = fmin(1.0/sqrt(dr00), big);
					dr01 = fmin(1.0/sqrt(dr01), big);
					dr10 = fmin(1.0/sqrt(dr10), big);
					dr11 = fmin(1.0/sqrt(dr11), big);
					double drtotal = dr00 + dr01 + dr10 + dr11;


					double a_r_t0 = GasAcc_h[(ip - 1) * def_Gasnz_p + jp0].x * dr00 + GasAcc_h[(ip - 1) * def_Gasnz_p + jp0 + 1].x * dr01 + GasAcc_h[ip * def_Gasnz_p + jp1].x * dr10 + GasAcc_h[ip * def_Gasnz_p + jp1 + 1].x * dr11;
					double a_z_t0 = GasAcc_h[(ip - 1) * def_Gasnz_p + jp0].y * dr00 + GasAcc_h[(ip - 1) * def_Gasnz_p + jp0 + 1].y * dr01 + GasAcc_h[ip * def_Gasnz_p + jp1].y * dr10 + GasAcc_h[ip * def_Gasnz_p + jp1 + 1].y * dr11;
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
			a_x += (x4.x * a_r - x4.y * a_th) / r1;
			a_y += (x4.y * a_r + x4.x * a_th) / r1;

		}//end if ir

		if(x4.w >= 0.0){
			double v2 = v4.x * v4.x + v4.y * v4.y + v4.z * v4.z;
			//Kick
			v4.x += a_x * dt;
			v4.y += a_y * dt;
			v4.z += a_z * dt;
//printf("Gas 2 %d %g %g %g %g %g %g %g\n", id, v4.x, v4.y, v4.z, a_x, a_y, a_z, x4.w);
			double v2B = v4.x * v4.x + v4.y * v4.y + v4.z * v4.z;

			v4_h[id] = v4;
	
			if(x4.w > 0.0){
				U = 0.5 * x4.w * (v2 - v2B);
				EnergySum_h[id] += U;
			}

		}
	}
}

// *************************************************
// This kernel uses a file to read the gas disk structure
//
// ****************************************************
void GasAcc2_cpu(double4 *x4_h, double4 *v4_h, int *index_h, double *time_h, double2 *Msun_h, double *dt_h, const int N, double *EnergySum_h, const int Nst, const double Ct, const int nr, double2 GasDatatime, double4 *GasData_h, int UsegasPotential, int UsegasEnhance, int UsegasDrag, int UsegasTidalDamping, const double Mgiant, const int Nstart){

	int id = 0 * 1 + 0 + Nstart;

	double rin = 0.25;
	double rout = 1000.0;

	double G_alpha =     Gas_parameters_c[1];
	//double G_beta =      Gas_parameters_c[2];

	for(id = 0 * 1 + 0 + Nstart; id < N + Nstart; ++id){

		int st = 0;
		if(Nst > 1 && id < N + Nstart) st = index_h[id] / def_MaxIndex;

		double rs = log(rout / rin) / ((double)(nr - 1)); //slope in distance profile 

		double dt = dt_h[st] * Ct;
		double dTime = time_h[st] / 365.25;

//if(id == 0) printf("rs %g %g %g %g\n", rs, GasDatatime.x, GasDatatime.y, dTime);

		double Msun = Msun_h[st].x;
		double U = 0.0;

		double3 v_rel3;
		double v_rel_r, v_rel_th;

		double4 x4 = x4_h[id];
		double4 v4 = v4_h[id];

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

			double4 GasData0 = GasData_h[ri];
			double4 GasData1 = GasData_h[ri + 1];

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

			v4_h[id] = v4;
			if(x4.w > 0.0){
				U = 0.5 * x4.w * (v2 - v2B);
				EnergySum_h[id] += U;
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


// ************************************
// this kernel sums up all the Energy loss due to the Gas Disc and adds to the internal Energy
//
// It works for the case of multiple warps, but only 1 thread block
//
// Author: Simon Grimm
// March 2021
// *************************************


// ************************************
// this kernel sums up all the Energy loss due to the Gas Disc and adds to the internal Energy
//
//It works for the case of only 1 single warp
//
// Author: Simon Grimm
// March 2021
// *************************************

#if def_CPU == 1
void gasEnergy_cpu(double *EnergySum_h, double *U_h, const int N){

	double U = 0.0;

	for(int id = 0; id < N; ++id){
		U += EnergySum_h[id];
		EnergySum_h[id] = 0.0;
	}
//printf("%d %g\n", idy, U_s[idy]);
	U_h[0] += U;
//printf("Uc %.20g %.20g\n", U, U_h[0]);
}
#endif


// ************************************
// This kernel sums up all the Energy loss due to the Gas Disc and adds to the internal Energy
// Uses paralel reduction sum with warp shuffle operations
// Author: Simon Grimm
// February 2023
// *************************************

//This function calls the Gas Energy kernel
void Data::gasEnergyCall(){
	int NN = N_h[0] + Nsmall_h[0];
#if def_CPU == 0
	if(NN <= WarpSize){
		gasEnergyc_kernel <<< 1, WarpSize, 0, hstream[0] >>> (EnergySum_h, U_h, NN);
	}
	else if(NN <= 512){
		int nn = (NN + WarpSize - 1) / WarpSize;
		gasEnergya_kernel <<< 1, nn * WarpSize, WarpSize * sizeof(double), hstream[0] >>> (EnergySum_h, U_h, NN);
	}
	else{
		int nct = 512;
		int ncb = min((NN + nct - 1) / nct, 1024);
		gasEnergyd1_kernel <<< ncb, nct, WarpSize * sizeof(double), hstream[0] >>> (EnergySum_h, vold_h, NN);
		gasEnergyd2_kernel <<< 1, ((ncb + WarpSize - 1) / WarpSize) * WarpSize, WarpSize * sizeof(double), hstream[0] >>> (U_h, vold_h, ncb);
	}
#else
	gasEnergy_cpu (EnergySum_h, U_h, NN);

#endif
}

void Data::GasAccCall(double *time_h, double *dt_h, double Ct){
	int nt = min(32, NB[0]);
	GasAcc_cpu /* (N_h[0] + nt - 1) / nt , nt */ (x4_h, v4_h, index_h, GasDisk_h, GasAcc_h, time_h, Msun_h, dt_h, N_h[0], EnergySum_h, Nst, Ct, P.UsegasPotential, P.UsegasEnhance, P.UsegasDrag, P.UsegasTidalDamping, P.G_Mgiant, 0, G_Nr_p);
}
void Data::GasAccCall_small(double *time_h, double *dt_h, double Ct){
	if(Nsmall_h[0] > 0) GasAcc_cpu /*(Nsmall_h[0] + 127)/128, 128 */ (x4_h + N_h[0], v4_h + N_h[0], index_h + N_h[0], GasDisk_h, GasAcc_h, time_h, Msun_h, dt_h, Nsmall_h[0], EnergySum_h, Nst, Ct, P.UsegasPotential, P.UsegasEnhance, P.UsegasDrag, P.UsegasTidalDamping, P.G_Mgiant, 0, G_Nr_p);
}
void Data::GasAccCall2_small(double *time_h, double *dt_h, double Ct){
	if(Nsmall_h[0] > 0) GasAcc2_cpu /*(Nsmall_h[0] + 127)/128, 128 */ (x4_h + N_h[0], v4_h + N_h[0], index_h + N_h[0], time_h, Msun_h, dt_h, Nsmall_h[0], EnergySum_h, Nst, Ct, GasDatanr, GasDatatime, GasData_h, P.UsegasPotential, P.UsegasEnhance, P.UsegasDrag, P.UsegasTidalDamping, P.G_Mgiant, 0);
}

int Data::freeGas(){
	int error;


#if def_CPU ==1
	free(Gas_rg_h);
	free(Gas_zg_h);
	free(Gas_rho_h);
	free(GasDisk_h);
	free(GasAcc_h);

#endif
	error = 0;
	if(error != 0){
		printf("Cuda Gas free error = %d = %s\n",error, "-");
		fprintf(masterfile, "Cuda Gas free error = %d = %s\n",error, "-");
		return 0;
	}
	return 1;

}

