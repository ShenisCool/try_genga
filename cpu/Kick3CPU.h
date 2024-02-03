#ifndef KICK_H
#define KICK_H
#include "define.h"

// **************************************
// This function computes the term a = mi/rij^3 * Kij
// ****************************************
void accA(double3 &ac, double4 &x4i, double4 &x4j, double rcritvi, double rcritvj, int j, int i){
	if( i != j && x4i.w >= 0.0 && x4j.w > 0.0){
		double rsq, ir, ir3, s;
		double3 r3ij;
		double rcritv, rcritv2;
		double y, yy;

		r3ij.x = x4j.x - x4i.x;
		r3ij.y = x4j.y - x4i.y;
		r3ij.z = x4j.z - x4i.z;

		rsq = (r3ij.x*r3ij.x) + (r3ij.y*r3ij.y) + (r3ij.z*r3ij.z);
		rcritv = fmax(rcritvi, rcritvj);
		rcritv2 = rcritv * rcritv;

		ir = 1.0/sqrt(rsq);
		ir3 = ir*ir*ir;

		if(rsq >= 1.0 * rcritv2){
			s = x4j.w * ir3;
//if(i == 13723) printf("acc %d %d %.40g %.40g %.40g %.40g %.40g\n", i, j, rsq, ac.z, 0.0, s, x4j.w);
		}
		else{
			if(rsq <= 0.01 * rcritv2){
				s = 0.0;
//if(i == 13723) printf("acc %d %d %.40g %.40g %.40g %.40g %.40g\n", i, j, rsq, ac.z, 0.0, s, x4j.w);
			}
			else{
				y = (rsq * ir - 0.1 * rcritv)/(0.9*rcritv);
				yy = y * y;
				s = (ir3 * yy) / (2.0*yy - 2.0*y + 1.0) * x4j.w;
//if(i == 13723) printf("acc %d %d %.40g %.40g %.40g %.40g %.40g\n", i, j, rsq, ac.z, y, s, x4j.w);
			}
		}
		ac.x += (r3ij.x * s);
		ac.y += (r3ij.y * s);
		ac.z += (r3ij.z * s);
	}
}

//**************************************
//This function computes the terms a = mi/rij^3 * Kij and b = mi/rij.
//This function also finds the pairs of bodies which are separated less than pc * rcritv^2. The index of those 
//pairs are stored in the array Encpairs_h in two different ways. This indexes are then used
//in the KickA32 kernel and in the Encounter kernel.
//
//E = 0: a + b + precheck (initial step)
//E = 1: a + b + precheck
//E = 2: a + precheck
//E = 10: a + b + precheck. (initial step) used for Test Particle Mode
//E = 11: a + b + precheck. used for Test Particle Mode
//E = 12: a + precheck. used for Test Particle Mode

//Authors: Simon Grimm
//August 2016
//****************************************
//float version

// ******************************************************
// Version of acc which is called from the recursive symplectic sub step method
// Author: Simon Grimm
// Janury 2019
// ******************************************************
void accS(double4 x4i, double4 x4j, double3 &ac, double *rcritv_h, int &NencpairsI, int2 *Encpairs2_h, const int i, const int j, const int NconstT, const int NencMax, const int SLevel, const int SLevels, const int E){

	if(i != j){

		double3 r3;
		double rsq;
		double ir, ir3;
		double y, yy;
		double rcritv, rcritv2;
		double rcritvi, rcritvj;

		r3.x = x4j.x - x4i.x;
		r3.y = x4j.y - x4i.y;
		r3.z = x4j.z - x4i.z;

		rsq = r3.x*r3.x + r3.y*r3.y + r3.z*r3.z + 1.0e-30;
		ir = 1.0/sqrt(rsq);
		ir3 = ir * ir * ir;

		double s = x4j.w * ir3 * def_ksq;

		for(int l = 0; l < SLevel; ++l){
		// (1 - K) factors of the previous levels 
//if(i == 0) printf(" (1-K%d)  ",l);
			rcritvi = rcritv_h[i + NconstT * l];
			rcritvj = rcritv_h[j + NconstT * l];

			rcritv = fmax(rcritvi, rcritvj);
			rcritv2 = rcritv * rcritv;

			if(rsq < 1.0 * rcritv2){
				if(rsq <= 0.01 * rcritv2){
					s *= 1.0;
				}
				else{
					y = (rsq * ir - 0.1 * rcritv)/(0.9*rcritv);
					yy = y * y;
					s *= (1.0 - yy / (2.0*yy - 2.0*y + 1.0));
				}
			}
			else s = 0.0;
		}

		
		if(SLevel < SLevels){
		//if(SLevel < SLevels - 1){ //<- use that for a complete last level Kick without BS
		// K factor of the current level
//if(i == 0) printf(" K%d  ",SLevel);
			rcritvi = rcritv_h[i + NconstT * SLevel];
			rcritvj = rcritv_h[j + NconstT * SLevel];

			rcritv = fmax(rcritvi, rcritvj);

			rcritv2 = rcritv * rcritv;


			if(rsq >= 1.0 * rcritv2){
				s *= 1.0;
			}
			else{
				if(rsq <= 0.01 * rcritv2){
					s = 0.0;

				}
				else{
					y = (rsq * ir - 0.1 * rcritv)/(0.9*rcritv);
					yy = y * y;
					s *= yy / (2.0*yy - 2.0*y + 1.0);

				}
			}
			//prechecker
			if(E == 0 || E == 2){	
				if(rsq < def_pc * rcritv2){	//prechecker
//printf("Precheck %d %d\n", i, j);
					Encpairs2_h[NencMax * i + NencpairsI].y = j;
					++NencpairsI;
				}
			}
		}

//if(i == 0 && j == 1) printf("\n");
		ac.x += (r3.x * s);
		ac.y += (r3.y * s);
		ac.z += (r3.z * s);
//printf("%.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g\n", x4i.w, x4i.x, x4i.y, x4i.z, x4j.w, x4j.x, x4j.y, x4j.z);
	}
}

// *********************************
//Only here for testing
//
//Authors: Simon Grimm, Joachim Stadel
////March 2014
//
// ********************************

// **************************************
//This kernel performs the first kick of the time step, in the case of no close encounters.
//It reuses the values from the second kick in the previous time step.

//Authors: Simon Grimm
//November 2016
// ****************************************

// **************************************
//This kernel performs the kick for a backup step.
//It reuses the values from the kick in the original time step.

//Authors: Simon Grimm
//November 2016
// ****************************************
void kick32C_cpu(double4 *x4_h, double4 *v4_h, double3 *ab_h, const int N, double dtksq){

	int id = 0 * 1 + 0;
	for(id = 0 * 1 + 0; id < N; ++id){
		double3 a = ab_h[id];
		if(x4_h[id].w >= 0.0){
			v4_h[id].x += (a.x * dtksq);
			v4_h[id].y += (a.y * dtksq);
			v4_h[id].z += (a.z * dtksq);
//printf("KickB %d %g %g %g %g\n", id, acck_h[id].x, acck_h[id].y, acck_h[id].z, v4_h[id].x);
		}
	}
}


// *******************************************
//This kernel is used to sort the close encounter list, to be able to reproduce simulations exactly
//It shoud be used only for debugging or special cases.

//Authors: Simon Grimm
//August 2016
// *********************************************
void Sortb_cpu(int2 *Encpairs2_h, const int Nstart, const int N, const int NencMax){

	int id = 0 * 1 + 0 + Nstart;	

	for(id = 0 * 1 + 0 + Nstart; id < N; ++id){
		int NI = Encpairs2_h[id * NencMax].x;
		NI = min(NI, NencMax);

		int stop = 0;
		while(stop == 0){
			stop = 1;
			for(int i = 0; i < NI - 1; ++i){
				int jj = Encpairs2_h[id * NencMax + i].y;
				int jjnext = Encpairs2_h[id * NencMax + i + 1].y;
//if(id == 13723) printf("sort %d %d %d %d\n", id, NI, jj, jjnext);
				if(jjnext < jj){
					//swap
					Encpairs2_h[id * NencMax + i].y = jjnext;
					Encpairs2_h[id * NencMax + i + 1].y = jj;
					stop = 0;

				}
			}
		}
		stop = 0;
	}
}

void SortSb_cpu(int *Encpairs3_h, int *Nencpairs3_h, const int N, const int NencMax){

	int idd = 0 * 1 + 0;

	for(idd = 0 * 1 + 0; idd < Nencpairs3_h[0]; ++idd){
		int id = Encpairs3_h[idd * NencMax + 1];
		if(id >= 0 && id < N){
			int NI = Encpairs3_h[id * NencMax + 2];
			NI = min(NI, NencMax);

			int stop = 0;
			while(stop == 0){
				stop = 1;
				for(int i = 0; i < NI - 1; ++i){
					int jj = Encpairs3_h[id * NencMax + i + 4];
					int jjnext = Encpairs3_h[id * NencMax + i + 1 + 4];
//printf("sortSb %d %d %d %d\n", id, NI, jj, jjnext);
					if(jjnext < jj){
						//swap
						Encpairs3_h[id * NencMax + i + 4] = jjnext;
						Encpairs3_h[id * NencMax + i + 1 + 4] = jj;
						stop = 0;

					}
				}
			}
		}
	}
}



// **************************************
//This kernel performs the first kick of the time step, in the case of close interactions.
//It reuses the values from the second kick in the previous time step, and adds the terms aij*dt*Kij for all
//the bodies involved in a close encounter.
//NI is the number of bodies involved in a close encounter with body i 

//Authors: Simon Grimm
//December 2016

// EE = 0  Do not update velocities
// EE >= 1, Do update velocities
// ****************************************
void kick32Ab_cpu(double4 *x4_h, double4 *v4_h, double3 *acck_h, double3 *ab_h, double *rcritv_h, const double dtksq, int *Nencpairs_h, int2 *Encpairs2_h, const int Nstart, const int N, const int NencMax, const int EE){

	int id = 0 * 1 + 0 + Nstart;	

	for(id = 0 * 1 + 0 + Nstart; id < N; ++id){
		double3 a = {0.0, 0.0, 0.0};
		double4 x4i = x4_h[id];
		if(x4i.w >= 0.0){
			if(Nencpairs_h[0] > 0){
				int NI = Encpairs2_h[id * NencMax].x;
				NI = min(NI, NencMax);
//if(NI > 0) printf("NI %d %d\n", id, NI);
				double rcritvi = rcritv_h[id];
				for(int i = 0; i < NI; ++i){
					int jj = Encpairs2_h[id * NencMax + i].y;
					double4 x4j = x4_h[jj];
//printf("AI %d %d %d %.40g %.40g %.40g %.20g %.20g %.40g\n", id, jj, NI, x4i.x, x4j.x, v4_h[id].z, x4j.x, x4j.w, a.z);
					double rcritvj = rcritv_h[jj];
					accA(a, x4i, x4j, rcritvi, rcritvj, jj, id);
				}
			
				double3 aa;
				aa.x = a.x + acck_h[id].x;
				aa.y = a.y + acck_h[id].y;
				aa.z = a.z + acck_h[id].z;

				if(EE >= 1){
					v4_h[id].x += (aa.x * dtksq);
					v4_h[id].y += (aa.y * dtksq);
					v4_h[id].z += (aa.z * dtksq);
				}
				ab_h[id] = aa;
			}
			else{
			
				double3 a = acck_h[id];
				if(EE >= 1){
					v4_h[id].x += (a.x * dtksq);
					v4_h[id].y += (a.y * dtksq);
					v4_h[id].z += (a.z * dtksq);
				}
//printf("KickB %d %.16e %.16e %.16e %.16e %.16e %.16e\n", id, acck_h[id].x, acck_h[id].y, acck_h[id].z, v4_h[id].x * dayUnit, v4_h[id].y * dayUnit, v4_h[id].z * dayUnit);
				ab_h[id] = a;
			}

		}
//if(id == 50) printf("K %d %.40g %.40g %.40g %.20g %.20g %.20g %.20g\n", id, v4_h[id].x, v4_h[id].y, v4_h[id].z, a.x, a.y, acck_h[id].x, acck_h[id].y);
	}
}

// *****************************************************
// This kernel collects the Encpairs2 information from multiple GPUs into the main array.
//
// Author: Simon Grimm
// November 2022
// ********************************************************

// *****************************************************
// Version of the Kick kernel which is called from the recursive symplectic sub step method
//
// Author: Simon Grimm
// January 2019
// ********************************************************
void kickS_cpu(double4 *x4_h, double4 *v4_h, double4 *xold_h, double4 *vold_h, double *rcritv_h, const double dtksq, int *Nencpairs_h, int2 *Encpairs_h, int2 *Encpairs2_h, int *Nencpairs3_h, int *Encpairs3_h, const int N, const int NconstT, const int NencMax, const int SLevel, const int SLevels, const int E){

	int idd = 0 * 1 + 0;	

	#pragma omp parallel for
	for(idd = 0 * 1 + 0; idd < Nencpairs3_h[0]; ++idd){
		double3 a = {0.0, 0.0, 0.0};
		int id = Encpairs3_h[idd * NencMax + 1];

		if(id >= 0 && id < N){
			double4 x4i = xold_h[id];
			double4 v4i = vold_h[id];
			int NencpairsI = 0;
			if(x4i.w >= 0.0){
				int NI = Encpairs3_h[id * NencMax + 2];
				NI = min(NI, NencMax);
//if(NI > 0) printf("NI %d %d %d\n", idd, id, NI);
				for(int i = 0; i < NI; ++i){
					int jj = Encpairs3_h[id * NencMax + i + 4];
					double4 x4j = xold_h[jj];
					if(x4j.w >= 0.0){
//if(E == 0) printf("AI %d %d %d %d %.40g %.40g %.40g %.40g\n", idd, id, jj, NI, x4i.x, x4j.x, v4_h[id].z, a.z);
						accS(x4i, x4j, a, rcritv_h, NencpairsI, Encpairs2_h, id, jj, NconstT, NencMax, SLevel, SLevels, E);
					}
				}
				double3 aa;
				aa.x = (a.x * dtksq);
				aa.y = (a.y * dtksq);
				aa.z = (a.z * dtksq);

				v4i.x += aa.x;
				v4i.y += aa.y;
				v4i.z += aa.z;

				if(E == 0){
					x4_h[id] = x4i;
				}
				v4_h[id] = v4i;
//printf("KickS %d %d %.20g %.20g %.20g\n", idd, id, v4_h[id].x, v4_h[id].y, v4_h[id].z);
				if(E == 0 || E == 2){
					for(int i = 0; i < NencpairsI; ++i){
						int jj = Encpairs2_h[id * NencMax + i].y;
						if(id > jj){
#if def_CPU == 0
							int Ne = atomicAdd(Nencpairs_h, 1);
#else
							int Ne;
							#pragma omp atomic capture
							Ne = Nencpairs_h[0]++;
#endif
//printf("KickS %d %d %d\n", Ne, id, jj);
							Encpairs_h[Ne].x = id;
							Encpairs_h[Ne].y = jj;
						}
					}
				}
			}
		}
	}
}


// **************************************
//This kernel performs the first kick of the time step, in the case of close interactions.
//It reuses the values from the second kick in the previous time step, and adds the terms aij*dt*Kij for all
//the bodies involved in a close encounter.
//NI is the number of bodies involved in a close encounter with body i 
//It checks if a transit occurs in the next time step

//Authors: Simon Grimm
//December 2016
// ****************************************

// **************************************
//This kernel performs the second kick of the time step, in the case NB = 16. NB is the next bigger number of N
//which is a power of two.
//It calculates the acceleration between all bodies with respect to the changeover function K.
//It also calculates all accelerations from bodies not beeing in a close encounter and store it in accK_h. This values will then be used 
//it the next time step.
//It performs also a precheck for close encouter candidates. This pairs are stored in the array Encpairs_h.
//The number of close encounter candidates is stored in Nencpairs_h.
//
//E = 0: Precheck + acck. used in initial step
//E = 1: Kick + Precheck + acck. used in main steps
//E = 2: Kick + Precheck. used in mid term steps of higher order integration
//
//The Kernel is launched with N blocks a NB theads.
//
//Authors: Simon Grimm
//April 2019
//****************************************
//float Version

// **************************************
//This kernel performs the second kick of the time step, in the case 32 <= NB < 128. NB is the next bigger number of N
//which is a power of two.
//It calculates the acceleration between all bodies with respect to the changeover function K.
//It also calculates all accelerations from bodies not beeing in a close encounter. This values will then be used 
//it the next time step.
//It performs also a precheck for close encouter candidates. This pairs are stored in the array Encpairs_h.
//The number of close encounter candidates is stored in Nencpairs_h.
//
//E = 0: Precheck + acck. used in initial step
//E = 1: Kick + Precheck + acck. used in main steps
//E = 2: Kick + Precheck. used in mid term steps of higher order integration*
//
//The Kernel is launched with N blocks a NB theads.

//Authors: Simon Grimm
//April 2019
//
//****************************************
//float Version

// **************************************
//This kernel performs the second kick of the time step.
//It calculates the acceleration between all bodies with respect to the changeover function K.
//It also calculates all accelerations from bodies not beeing in a close encounter and store it in accK_h. This values will then be used 
//it the next time step.
//It performs also a precheck for close encouter candidates. This pairs are stored in the array Encpairs_h.
//The number of close encounter candidates is stored in Nencpairs_h.
//NT is the total number of bodies, Nstart is the starting index
//
//E = 0: Precheck + acck. used in initial step
//E = 1: Kick + Precheck + acck. used in main steps
//E = 2: Kick + Precheck. used in mid term steps of higher order integration
//
//Authors: Simon Grimm, Joachim Stadel
////March 2014
//
// ****************************************


// CC = 1, do pre-check
// CC = 0, don't do pre-check
#endif
