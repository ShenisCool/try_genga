#ifndef FG2CPU_H
#define FG2CPU_H

#include "Orbit2CPU.h"
#include "BSSingleCPU.h"

float Gridae_c[9];
int GridaeN_c[3];
double S_c[def_FGN + 1];
double C_c[def_FGN + 1];
int UseaeGrid_c[1];


//**************************************
// This function copies the aeGrid parameters to constant memory. This functions must be in
// the same file as the use of the constant memory
//
//Authors: Simon Grimm
//April 2015
//
//***************************************/
__host__ void Data::constantCopy(){
	float GridaeP[9] = {Gridae.amin, Gridae.amax, Gridae.emin, Gridae.emax, Gridae.imin, Gridae.imax, Gridae.deltaa, Gridae.deltae, Gridae.deltai};
	int GridaeN[3] = {Gridae.Na, Gridae.Ne, Gridae.Ni};
#if def_CPU == 0
	cudaMemcpyToSymbol(Gridae_c, GridaeP, 9*sizeof(float), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(GridaeN_c, GridaeN, 3*sizeof(int), 0, cudaMemcpyHostToDevice);
#else
	memcpy(Gridae_c, GridaeP, 9*sizeof(float));
	memcpy(GridaeN_c, GridaeN, 3*sizeof(int));
#endif
}
//**************************************
// This function copies the use ae grid flag to constant memory. This functions must be in
// the same file as the use of the constant memory
//
//Authors: Simon Grimm
//Mai 2015
//
//***************************************/
__host__ void Data::constantCopy2(){

#if def_CPU == 0
	cudaMemcpyToSymbol(UseaeGrid_c, &P.UseaeGrid, sizeof(int), 0, cudaMemcpyHostToDevice);
#else
	memcpy(UseaeGrid_c, &P.UseaeGrid, sizeof(int));
#endif
}


__host__ void Data::constantCopySC(double *S_h, double *C_h){
#if def_CPU == 0
	cudaMemcpyToSymbol(S_c, S_h, (def_FGN + 1) * sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(C_c, C_h, (def_FGN + 1) * sizeof(double), 0, cudaMemcpyHostToDevice);
#else
	memcpy(S_c, S_h, (def_FGN + 1) * sizeof(double));
	memcpy(C_c, C_h, (def_FGN + 1) * sizeof(double));
#endif
}


//**************************************
//based on a code
//from Joachim Stadel
//See Danby for f and g method
//
//Authors: Simon Grimm, Joachim Stadel
//July 2016
//
//***************************************/
 void fgfull(double4 &x4i, double4 &v4i, double dt, const double mu, const double Msun, const float4 aelimits, unsigned int &aecount, unsigned int *Gridaecount_h, unsigned int *Gridaicount_h, const int si, const int id, const int index, const int UseGR){

	if(x4i.w >= 0.0){

		volatile double dec;                                      /* delta E */
		volatile double dm;
		double mw;                                       /* minus function to zero */
		double wp;                                       /* first derivative */
		volatile double wpp;                                      /* second derivative */
		volatile double wppp;                                     /* third derivative */
		volatile double dx;
		double s,c;
		const double DOUBLE_EPS = 1.2e-16;
		double converge;
		double UP = 2*M_PI;
		double LOW = -2*M_PI;
		int i;
		/*
		* Evaluate some orbital quantites.
		*/

		volatile double rsq = (x4i.x * x4i.x) + (x4i.y * x4i.y) + (x4i.z * x4i.z);
		volatile double vsq = (v4i.x * v4i.x) + (v4i.y * v4i.y) + (v4i.z * v4i.z);
		double u =  (x4i.x * v4i.x) + (x4i.y * v4i.y) + (x4i.z * v4i.z);
		volatile double ir = 1.0/sqrt(rsq);
		double ia = 2.0*ir-vsq/mu;

		if(UseGR == 1){// GR time rescale (Saha & Tremaine 1994)
			double c2 = def_cm * def_cm;
			dt *= 1.0 - 1.5 * mu * ia / c2;
		}
		if(ia > 0.0){
			double t1 = ia*ia;
			volatile double ria = rsq*ir*ia;
			volatile double en = sqrt(mu*t1*ia);
			double ien = 1.0/en;
			volatile double ec = 1.0-ria;
			volatile double es = u*t1*ien;
			volatile double e = sqrt(ec*ec + es*es);
			double a = 1.0/ia;
			if(UseaeGrid_c[0] == 1){
				int na = (int)((a - Gridae_c[0]) / Gridae_c[6]);
				int ne = (int)((e - Gridae_c[2]) / Gridae_c[7]);
				if(si == 0 && na >= 0 && na < GridaeN_c[0] && ne >= 0 && ne < GridaeN_c[1]){
#if def_CPU == 0
					atomicAdd(&Gridaecount_h[ne * GridaeN_c[0] + na], 1); 
#else
					#pragma omp atomic
					Gridaecount_h[ne * GridaeN_c[0] + na]++;
#endif
		
				}
			
				//compute inclination
				double inc;
				double3 h3;
				h3.x = x4i.y * v4i.z - x4i.z * v4i.y;
				h3.y = -x4i.x * v4i.z + x4i.z * v4i.x;
				h3.z = x4i.x * v4i.y - x4i.y * v4i.x;

				double h2 = h3.x * h3.x + h3.y * h3.y + h3.z * h3.z;
				double h = sqrt(h2);

				double t = h3.z / h;

				if(t <= -1){
					inc = M_PI;
				}
				else{
					if(t < 1){
						inc = acos(t);
					}
					else inc = 0.0;
				}
				int ni = (int)((inc - Gridae_c[4]) / Gridae_c[8]);

				if(si == 0 && na >= 0 && na < GridaeN_c[0] && ni >= 0 && ni < GridaeN_c[2]){
#if def_CPU == 0
					atomicAdd(&Gridaicount_h[ni * GridaeN_c[0] + na], 1); 
#else
					#pragma omp atomic
					Gridaicount_h[ni * GridaeN_c[0] + na]++;
#endif
				}
			}
			if(e >= aelimits.z && e <= aelimits.w){
				if(a >= aelimits.x && a <= aelimits.y){
					aecount = 1u;
				}
			}
			dm = en * dt - es;
			if((es*cos(dm)+ec*sin(dm)) > 0){
				dec = fma(0.85, e, dm); //dm + 0.85*e;
			}
			else dec = fma(-0.85, e, dm); //dm - 0.85*e;
			converge = fabs(en * dt *DOUBLE_EPS);

			for(i = 0; i < 128; ++i) {

				//s = sin(dec);
				//c = cos(dec);
				sincos(dec, &s, &c);
				wpp = ec*s + es*c;
				wppp = ec*c - es*s;
				mw = dm - dec + wpp;
				if(mw < 0.0){
					UP = dec;
				}
				else LOW = dec;
				wp = 1.0 - wppp;
				wpp *= 0.5;
				dx = mw/wp;
				dx = mw/(wp + dx*wpp);
				dx = mw/(wp + dx*(wpp + (1.0/6.0)*dx*wppp));
				double next = dec + dx;
				if (fabs(dx) <= converge) break;
				if(next > LOW && next < UP){
					dec = next;
				}
				else dec = 0.5*(LOW + UP);
				if (dec==LOW || dec==UP) break;
			}
			if(i < 127){
				double iwp = 1.0/wp;
				double air_ = -1.0/ria;
				double t1 = (1.0-c);
				double f = 1.0 + air_*t1;
				double g = dt + (s-dec)*ien;
				double fd = air_*iwp*s*en;
				double gd = 1.0 - iwp*t1;
				double tx = f*x4i.x+g*v4i.x;
				double ty = f*x4i.y+g*v4i.y;
				double tz = f*x4i.z+g*v4i.z;

				v4i.x = fd*x4i.x+gd*v4i.x;
				v4i.y = fd*x4i.y+gd*v4i.y;
				v4i.z = fd*x4i.z+gd*v4i.z;

				x4i.x = tx;
				x4i.y = ty;
				x4i.z = tz;

			}
			else{
				BSSinglestep(x4i, v4i, Msun, dt, id);
//printf("%d %g\n", id, 1.0/ia);
			}
		}
		else{
			BSSinglestep(x4i, v4i, Msun, dt, id);
//printf("%d %g\n", id, 1.0/ia);
		}
	}
}


//**************************************
//this functions is not jet fully working and not energy conserving
//
//based on a code
//from Joachim Stadel
//See Danby for f and g method
//
//Authors: Simon Grimm, Joachim Stadel
//July 2016
//
//***************************************/
void fastfg(double4 &x4i, double4 &v4i, double dt, const double mu, const double Msun, float4 aelimits, unsigned int &aecount, int *Gridaecount_h, const int si, const int id, const int UseGR){

	if(x4i.w >= 0.0){
		int ii,i,j,jnew;
		double sgn,dEj,f0,f1,f2,f3;
		double y,dy,y2,y4,A,B;

		double s,c,wp;
		double f = 0.0;	  // Gauss's f, g, fdot and gdot
		double g = 0.0;
		double fd = 0.0;
		double gd = 0.0;

		double rsq, ir,vsq;
		double u;		  // r v cos(phi)
		double ia;		  // semi-major axis
		double ec,es;		  // e cos(E), e sin(E)
		double ien, en;		  // mean motion
		double dM;		  // delta mean anomoly
		double t1;
		double ria, air_, iwp;
		double3 t;

		s = 0.0;
		c = 0.0;

		int ok = 1;
		
		rsq = x4i.x*x4i.x + x4i.y*x4i.y + x4i.z*x4i.z;
		vsq = v4i.x*v4i.x + v4i.y*v4i.y + v4i.z*v4i.z;
		u =  x4i.x*v4i.x + x4i.y*v4i.y + x4i.z*v4i.z;
		ir = 1.0/sqrt(rsq);
		ia = 2.0*ir-vsq/mu;
		if(UseGR == 1){// GR time rescale (Saha & Tremaine 1994)
			double c2 = def_cm * def_cm;
			dt *= 1.0 - 1.5 * mu * ia / c2;
		}

		if(ia > 0.0){

			t1 = ia*ia;
			ria = rsq*ir*ia;
			en = sqrt(mu*t1*ia);
			ien = 1.0/en;
			ec = 1.0-ria;
			es = u*t1*ien;
			dM = en * dt;

			dEj = dM;
			sgn = dEj < 0 ? -1.0 : 1.0;
			dM -= es;
			j = (int)floor(sgn*dEj*N_PI) + 1;

			for(i = 0; i < 6; ++i){  //32

				f2 = es*C_c[j] + sgn*ec*S_c[j];
				f3 = -sgn*es*S_c[j] + ec*C_c[j];
				dEj = sgn*j*PI_N;
				f0 = dEj - dM - f2;
				f1 = 1.0 - f3;
				y = -f0/f1;
				dEj += y;
				jnew = (int)floor(sgn*dEj*N_PI) + 1;
				if (jnew == j) break;
				j = jnew;
			}
			if(i >= 5) ok = 0; //31

			dEj = sgn*j*PI_N;
			y = (-f0 + y*y*(0.5*f2 + (1.0/3.0)*f3*y))/(f1 + y*(f2 + 0.5*f3*y));

			for(ii = 0; ii < 6; ++ii){ //32
				y2 = y*y;
				y4 = y2*y2;
				B = f2*y*(1.0 - (1.0/6.0)*y2 + (1.0/120.0)*y4) + 0.5*f3*y2*(1.0 - (1.0/12.0)*y2 + (1.0/360.0)*y4);
				A = 0.5*f2*y*(1.0 - 0.25*y2 + (1.0/72.0)*y4) + (1.0/3.0)*f3*y2*(1.0 - 0.1*y2 + (1.0/280.0)*y4);	
				dy = (-f0 + y*A)/(f1 + B) - y;
				if (fabs(dy) < 1e-20) break;
				y += dy;
			}
			if(ii >= 5) ok = 0; //31
			dEj += y;
			sincos(dEj, &s, &c);

			air_ = -1.0/ria;
			t1 = 1.0 - c;
			wp = 1.0 - ec*c + es*s;
			iwp = 1.0/wp;
			f = 1.0 + air_ * t1;
			g = dt + (s-dEj)*ien;
			fd = air_ * iwp * en * s;
			gd = 1.0 - t1*iwp;

		}
		if(ia <= 0 || ok == 0){
//	printf("%g %d %d", ia, i, ii);
			//fgfull(x4i, v4i, dt, def_ksq * Msun, Msun, aelimits, aecount, Gridaecount_h, si, id);
			BSSinglestep(x4i, v4i, Msun, dt, id);
		}
		else{
			t.x = f*x4i.x+g*v4i.x;
			t.y = f*x4i.y+g*v4i.y;
			t.z = f*x4i.z+g*v4i.z;
			v4i.x = fd*x4i.x+gd*v4i.x;
			v4i.y = fd*x4i.y+gd*v4i.y;
			v4i.z = fd*x4i.z+gd*v4i.z;
			x4i.x = t.x;
			x4i.y = t.y;
			x4i.z = t.z;
		}
	}
}
// ******************************************
//This function calculates the Poincare surcafe of section
//It markes paricles crossing the section and set the Flag PFlag_h
//Authors: Simon Grimm, Joachim Stadel
//March 2014
//
// ******************************************
void PoincareSection_cpu(double4 *x4_h, double4 *v4_h, double4 *xold_h, double4 *vold_h, int *index_h, const double Msun, const int N, const int si, int *PFlag_h){

	int id = 0 * 1 + 0;



	for(id = 0 * 1 + 0; id < N; ++id){
		if(si == 0){
			float4 aelimits = {0.0f, 0.0f, 0.0f, 0.0f};
			unsigned aecount = 0u;
			unsigned int Gridaecount = 0u;
			unsigned int Gridaicount = 0u;

			double4 x4i = x4_h[id];
			double4 x4oldi = xold_h[id];
			double4 v4oldi = vold_h[id];
			int index = index_h[id];
			if(x4oldi.y < 0.0 && x4i.y >= 0.0 && x4i.x > 0.0){
				PFlag_h[0] = 1;
				double dtt = -x4oldi.y / v4oldi.y;
				fgfull(x4oldi, v4oldi, dtt, def_ksq * Msun, Msun, aelimits, aecount, &Gridaecount, &Gridaicount, si, id, index, 0);
	//			printf("%g %g %g\n", x4oldi.x, x4oldi.y, v4oldi.x);
				xold_h[id] = x4oldi;
				vold_h[id] = v4oldi;
				vold_h[id].w *= -1.0;		//Flag particles
			}
		}
	}
}


// **************************************
//The fg_kernel does a copy of the coordinates and calls the FG function to perform the Kepler drift.
//There are 2 different FG, and one Burlish Stoer function, fastest one is fastfg.
//
//Author: Simon Grimm
//July 2016
//
// *****************************************
void fg_cpu(double4 *x4_h, double4 *v4_h, double4 *xold_h, double4 *vold_h, int *index_h, const double dt, const double Msun, const int N, float4 *aelimits_h, unsigned int *aecount_h, unsigned int *Gridaecount_h, unsigned int *Gridaicount_h, const int si, const int UseGR){

	int id = 0 * 1 + 0;

	#pragma omp parallel for
	for(id = 0 * 1 + 0; id < N; ++id){
		unsigned int aecount = 0u;
		double4 x4i = x4_h[id];
		double4 v4i = v4_h[id];
		xold_h[id] = x4i;
		vold_h[id] = v4i;
		int index = index_h[id];
//if(id < 10) printf("FGA %d %d %g %.20e %.20e %.20e %.20e %.20e %.20e e %.20e\n", id, index_h[id], x4_h[id].w, x4_h[id].x, x4_h[id].y, x4_h[id].z, v4_h[id].x, v4_h[id].y, v4_h[id].z, (x4_h[id].x * v4_h[id].x) + (x4_h[id].y * v4_h[id].y));
		float4 aelimits = aelimits_h[id];
		//fastfg(x4i, v4i, dt, def_ksq * Msun, Msun, aelimits, aecount, Gridaecount_h, si, id, UseGR);
		fgfull(x4i, v4i, dt, def_ksq * Msun, Msun, aelimits, aecount, Gridaecount_h, Gridaicount_h, si, id, index, UseGR);
		//BSSinglestep(x4i, v4i, Msun, dt, id); //GR not included here
		if(si >= 0){
			//dont update arrays during tunig process
			x4_h[id] = x4i;
			v4_h[id] = v4i;
		}
//if(id < 10)  printf("FGB %d %d %g %.20e %.20e %.20e %.20e %.20e %.20e e %.20e\n", id, index_h[id], x4_h[id].w, x4_h[id].x, x4_h[id].y, x4_h[id].z, v4_h[id].x, v4_h[id].y, v4_h[id].z, (x4_h[id].x * v4_h[id].x) + (x4_h[id].y * v4_h[id].y));
		if(si == 0){
			aecount_h[id] += aecount;
		}
	}
}



// *****************************************************
// Version of the FG kernel which is called from the recursive symplectic sub step method
// calls fg only if there are close encounter candidates in the current recursion level
//
// Author: Simon Grimm
// January 2019
// ********************************************************
void fgS_cpu(double4 *x4_h, double4 *v4_h, double4 *xold_h, double4 *vold_h, int *index_h, const double dt, const double Msun, const int N, float4 *aelimits_h, unsigned int *aecount_h, unsigned int *Gridaecount_h, unsigned int *Gridaicount_h, const int si, const int UseGR, int *Nencpairs3_h, int *Encpairs3_h, const int NencMax){

	int idd = 0 * 1 + 0;

	#pragma omp parallel for
	for(idd = 0 * 1 + 0; idd < Nencpairs3_h[0]; ++idd){
		int id = Encpairs3_h[idd * NencMax + 1];
		if(id >= 0 && id < N){
			unsigned int aecount = 0u;
			double4 x4i = x4_h[id];
			double4 v4i = v4_h[id];
			xold_h[id] = x4i;
			vold_h[id] = v4i;
			int index = index_h[id];
//printf("FGA %d %d %.20e %.20e %.20e %.20e %.20e %.20e e %.20e\n", idd, id, x4_h[id].x, x4_h[id].y, x4_h[id].z, v4_h[id].x, v4_h[id].y, v4_h[id].z, x4_h[id].x * v4_h[id].x + x4_h[id].y * v4_h[id].y);
			float4 aelimits = aelimits_h[id];
			//fastfg(x4i, v4i, dt, def_ksq * Msun, Msun, aelimits, aecount, Gridaecount_h, si, id, UseGR);
			fgfull(x4i, v4i, dt, def_ksq * Msun, Msun, aelimits, aecount, Gridaecount_h, Gridaicount_h, si, id, index, UseGR);
			//BSSinglestep(x4i, v4i, Msun, dt, id); //GR not included here
			x4_h[id] = x4i;
			v4_h[id] = v4i;
//printf("FGB %d %d %.20e %.20e %.20e %.20e %.20e %.20e e %.20e\n", idd, id, x4_h[id].x, x4_h[id].y, x4_h[id].z, v4_h[id].x, v4_h[id].y, v4_h[id].z, x4_h[id].x * v4_h[id].x + x4_h[id].y * v4_h[id].y);
			if(si == 0){
				aecount_h[id] += aecount;
			}
		}
	}
}

// **************************************
//for multi simulation mode
//The fg_kernel does a copy of the coordinates and calls the FG function to perform the Kepler drift.
//
//Authors: Simon Grimm
//July 2016
//
// *****************************************

#endif
