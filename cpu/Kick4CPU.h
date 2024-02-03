#ifndef KICK4_H
#define KICK4_H
#include "define.h"


//**************************************
//This function computes the terms a = mi/rij^3 * Kij.
//This function also finds the pairs of bodies which are separated less than pc * rcritv^2. 
//The function writes the encounter pairs into a list.

//Authors: Simon Grimm
//March 2019
//****************************************

//float version

// ********************************************************************************************
// This kernel sets all close Encounter lists to zero. It also sets the acceleration to zero.
// It is needed in the tunig loop for the kick kernel parameters
//
//Date: March 2019
//Author: Simon Grimm
// *******************************************************************************************
void EncpairsZeroC_cpu(int2 *Encpairs2_h, double3 *a_h, int *Nencpairs_h, int *Nencpairs2_h, const int NencMax, const int N){

	int id = 0 + 0 * 1;

	if(id == 0){
		Nencpairs_h[0] = 0;
		Nencpairs2_h[0] = 0;
	}

	for(id = 0 + 0 * 1; id < N; ++id){
		Encpairs2_h[NencMax * id].x = 0;

		a_h[id].x = 0.0;
		a_h[id].y = 0.0;
		a_h[id].z = 0.0;
	}
}

void compare_a_cpu(double3 *a_h, double3 *ab_h, const int KickFloat, const int N, const int f){

	int id = 0 + 0 * 1;

	for(id = 0 + 0 * 1; id < N; ++id){
		if(f == 1){
			double dx = fabs(a_h[id].x - ab_h[id].x);
			double dy = fabs(a_h[id].y - ab_h[id].y);
			double dz = fabs(a_h[id].z - ab_h[id].z);
//printf("compare a %d %.20g %.20g %.20g | %.20g %.20g %.20g\n", id, a_h[id].x, a_h[id].y, a_h[id].z, ab_h[id].x, ab_h[id].y, ab_h[id].z);
			if(KickFloat == 0){
				if(dx + dy + dz > 1.0e-8){
					printf("Comparison of acc from different kick kernel tuning parameters failed %d %.20g %.20g %.20g %.20g %.20g %.20g\n", id, a_h[id].x, ab_h[id].x, a_h[id].y, ab_h[id].y, a_h[id].z, ab_h[id].z);
				}
			}
			else{
				if(dx + dy + dz > 1.0e-6){
					printf("Comparison of acc from different kick kernel tuning parameters failed %d %.20g %.20g %.20g %.20g %.20g %.20g\n", id, a_h[id].x, ab_h[id].x, a_h[id].y, ab_h[id].y, a_h[id].z, ab_h[id].z);
				}
			}
		}
		ab_h[id] = a_h[id];
	}
}

// **********************************************************
// This kernel calculates the acceleration between all particles 
// the parallelization can be different according to the GPU one uses
// kernel parameters can be determinded beforehand in a tuning step

// The kernel writes a list of all close encounter candidates

// EE = 0: used for normal particles
// EE = 1: used for normal test particles
// EE = 2: used for semi test particles
// Date: April 2019
// Author: Simon Grimm

//float version
#if def_CPU == 1

// Can be applied only to massive particles
// Here EE is always 0
// Serial version
void Data::acc4E_cpu(){
	
	//if E == 0
	for(int i = 0; i < N_h[0]; ++i){
		a_h[i] = {0.0, 0.0, 0.0};
		Encpairs2_h[i * P.NencMax].x = 0;
	}
	
	for(int i = 0; i < N_h[0]; ++i){
		double ir, ir3;
		double3 r3ij;

		for(int j = i + 1; j < N_h[0]; ++j){
			
			if(x4_h[i].w >= 0.0 && x4_h[j].w >= 0.0){
				
//printf("%d %d %d\n", i, j);
				
				r3ij.x = x4_h[j].x - x4_h[i].x;
				r3ij.y = x4_h[j].y - x4_h[i].y;
				r3ij.z = x4_h[j].z - x4_h[i].z;
				
				double rsq = (r3ij.x*r3ij.x) + (r3ij.y*r3ij.y) + (r3ij.z*r3ij.z);
				
				double rcritv = fmax(rcritv_h[i], rcritv_h[j]);
				
				ir = 1.0/sqrt(rsq);
				ir3 = ir*ir*ir;
				
				
//				if(rsq < 3.0 * rcritv * rcritv){
				if(rsq < def_pc * rcritv * rcritv && (x4_h[i].w > 0.0 || x4_h[j].w > 0.0)){
					
					int Ni = Encpairs2_h[i * P.NencMax].x++;
					int Nj = Encpairs2_h[j * P.NencMax].x++;
//printf("enc1 %d %d %d %d\n", i, j, Ni, Nj);
					if(Ni >= P.NencMax){
						EncFlag_m[0] = max(EncFlag_m[0], Ni);
					}
					else{
						Encpairs2_h[i * P.NencMax + Ni].y = j;
					}

					if(Nj >= P.NencMax){
						EncFlag_m[0] = max(EncFlag_m[0], Nj);
					}
					else{
						Encpairs2_h[j * P.NencMax + Nj].y = i;
					}
					
					if(Ni < P.NencMax && Nj < P.NencMax){
						// i < j is always true
						int Ne = Nencpairs_h[0]++;
						Encpairs_h[Ne].x = i;
						Encpairs_h[Ne].y = j;
//printf("Precheck %d %d %d\n", i, j, Ne);
					}

					ir3 = 0.0;
				}
				else if(rsq == 0.0){
					ir3 = 0.0;
				}
				
				double si = (x4_h[i].w * ir3);
				double sj = (x4_h[j].w * ir3);
				
				a_h[i].x += r3ij.x * sj;
				a_h[i].y += r3ij.y * sj;
				a_h[i].z += r3ij.z * sj;
				
				a_h[j].x -= r3ij.x * si;
				a_h[j].y -= r3ij.y * si;
				a_h[j].z -= r3ij.z * si;
				
//if(i == 10) printf("acci %d %d %g %g\n", i, j, sj * r3ij.x, b_h[k * NconstT + i].x);
//if(j == 10) printf("accj %d %d %g %g\n", j, i, -si * r3ij.x, b_h[k * NconstT + j].x);
				
			}
		}
	}
}

// Is applied only to test particles particles
// Serial Version
void Data::acc4Esmall_cpu(){
	
	for(int i = N_h[0]; i < N_h[0] + Nsmall_h[0]; ++i){
		a_h[i] = {0.0, 0.0, 0.0};
		Encpairs2_h[i * P.NencMax].x = 0;
	}
	
	for(int i = N_h[0]; i < N_h[0] + Nsmall_h[0]; ++i){
		double ir, ir3;
		double3 r3ij;

		for(int j = 0; j < N_h[0]; ++j){
			
			if(x4_h[i].w >= 0.0 && x4_h[j].w >= 0.0){
				
				
				r3ij.x = x4_h[j].x - x4_h[i].x;
				r3ij.y = x4_h[j].y - x4_h[i].y;
				r3ij.z = x4_h[j].z - x4_h[i].z;
				
				double rsq = (r3ij.x*r3ij.x) + (r3ij.y*r3ij.y) + (r3ij.z*r3ij.z);
				
				double rcritv = fmax(rcritv_h[i], rcritv_h[j]);
				
				ir = 1.0/sqrt(rsq);
				ir3 = ir*ir*ir;
				
				
//				if(rsq < 3.0 * rcritv * rcritv){
				if(rsq < def_pc * rcritv * rcritv && (x4_h[i].w > 0.0 || x4_h[j].w > 0.0)){
					
					int Ni = Encpairs2_h[i * P.NencMax].x++;
					int Nj = 0;
					if(Ni >= P.NencMax){
						EncFlag_m[0] = max(EncFlag_m[0], Ni);
					}
					else{
						Encpairs2_h[i * P.NencMax + Ni].y = j;
					}

					if(P.UseTestParticles == 2){
						Nj = Encpairs2_h[j * P.NencMax].x++;

						if(Nj >= P.NencMax){
							EncFlag_m[0] = max(EncFlag_m[0], Nj);
						}
						else{
							Encpairs2_h[j * P.NencMax + Nj].y = i;
						}
					}					
//printf("enc1 small %d %d %d\n", i, j, Ni);
//printf("enc1 %d %d %d %d\n", i, j, Ni, Nj);

					if(Ni < P.NencMax && Nj < P.NencMax){
						// j < i is always true
						int Ne = Nencpairs_h[0]++;
						Encpairs_h[Ne].x = i;
						Encpairs_h[Ne].y = j;
//printf("Precheck small %d %d %d\n", i, j, Ne);
					}

					ir3 = 0.0;
				}
				else if(rsq == 0.0){
					ir3 = 0.0;
				}
				
				double sj = (x4_h[j].w * ir3);
				
				a_h[i].x += r3ij.x * sj;
				a_h[i].y += r3ij.y * sj;
				a_h[i].z += r3ij.z * sj;
	
				if(P.UseTestParticles == 2){
					double si = (x4_h[i].w * ir3);
				
					a_h[j].x -= r3ij.x * si;
					a_h[j].y -= r3ij.y * si;
					a_h[j].z -= r3ij.z * si;
				
				}
//if(i == 10) printf("acci %d %d %g %g\n", i, j, sj * r3ij.x, b_h[k * NconstT + i].x);
//if(j == 10) printf("accj %d %d %g %g\n", j, i, -si * r3ij.x, b_h[k * NconstT + j].x);
				
			}
		}
	}
}

//float version
void Data::acc4Ef_cpu(){
	
	//if E == 0
	for(int i = 0; i < N_h[0]; ++i){
		a_h[i] = {0.0, 0.0, 0.0};
		Encpairs2_h[i * P.NencMax].x = 0;
	}
	
	for(int i = 0; i < N_h[0]; ++i){
		float ir, ir3;
		float3 r3ij;

		for(int j = i + 1; j < N_h[0]; ++j){
			
			if(x4_h[i].w >= 0.0 && x4_h[j].w >= 0.0){
				
//printf("%d %d %d %d %d\n", 0, ii, jj, i, j);
				
				r3ij.x = x4_h[j].x - x4_h[i].x;
				r3ij.y = x4_h[j].y - x4_h[i].y;
				r3ij.z = x4_h[j].z - x4_h[i].z;
				
				float rsq = (r3ij.x*r3ij.x) + (r3ij.y*r3ij.y) + (r3ij.z*r3ij.z);
				
				float rcritv = fmax(rcritv_h[i], rcritv_h[j]);
				
				ir = 1.0f/sqrt(rsq);
				ir3 = ir*ir*ir;
				
				
//				if(rsq < 3.0 * rcritv * rcritv){
				if(rsq < def_pc * rcritv * rcritv && (x4_h[i].w > 0.0 || x4_h[j].w > 0.0)){
					
					int Ni = Encpairs2_h[i * P.NencMax].x++;
					int Nj = Encpairs2_h[j * P.NencMax].x++;
//printf("enc1 %d %d %d %d\n", i, j, Ni, Nj);
					if(Ni >= P.NencMax){
						EncFlag_m[0] = max(EncFlag_m[0], Ni);
					}
					else{
						Encpairs2_h[i * P.NencMax + Ni].y = j;
					}

					if(Nj >= P.NencMax){
						EncFlag_m[0] = max(EncFlag_m[0], Nj);
					}
					else{
						Encpairs2_h[j * P.NencMax + Nj].y = i;
					}
					
					if(Ni < P.NencMax && Nj < P.NencMax){
						// i < j is always true
						int Ne = Nencpairs_h[0]++;
						Encpairs_h[Ne].x = i;
						Encpairs_h[Ne].y = j;
//printf("Precheck %d %d %d\n", i, j, Ne);
					}

					ir3 = 0.0f;
				}
				else if(rsq == 0.0f){
					ir3 = 0.0f;
				}
				
				double si = (x4_h[i].w * ir3);
				double sj = (x4_h[j].w * ir3);
				
				a_h[i].x += r3ij.x * sj;
				a_h[i].y += r3ij.y * sj;
				a_h[i].z += r3ij.z * sj;
				
				a_h[j].x -= r3ij.x * si;
				a_h[j].y -= r3ij.y * si;
				a_h[j].z -= r3ij.z * si;
				
//if(i == 10) printf("acci %d %d %g %g\n", i, j, sj * r3ij.x, b_h[k * NconstT + i].x);
//if(j == 10) printf("accj %d %d %g %g\n", j, i, -si * r3ij.x, b_h[k * NconstT + j].x);
				
			}
		}
	}
}
// Is applied only to test particles particles
// Serial Version
void Data::acc4Efsmall_cpu(){
	
	for(int i = N_h[0]; i < N_h[0] + Nsmall_h[0]; ++i){
		a_h[i] = {0.0, 0.0, 0.0};
		Encpairs2_h[i * P.NencMax].x = 0;
	}
	
	for(int i = N_h[0]; i < N_h[0] + Nsmall_h[0]; ++i){
		float ir, ir3;
		float3 r3ij;

		for(int j = 0; j < N_h[0]; ++j){
			
			if(x4_h[i].w >= 0.0 && x4_h[j].w >= 0.0){
				
				
				r3ij.x = x4_h[j].x - x4_h[i].x;
				r3ij.y = x4_h[j].y - x4_h[i].y;
				r3ij.z = x4_h[j].z - x4_h[i].z;
				
				float rsq = (r3ij.x*r3ij.x) + (r3ij.y*r3ij.y) + (r3ij.z*r3ij.z);
				
				float rcritv = fmax(rcritv_h[i], rcritv_h[j]);
				
				ir = 1.0f/sqrt(rsq);
				ir3 = ir*ir*ir;
				
				
//				if(rsq < 3.0 * rcritv * rcritv){
				if(rsq < def_pc * rcritv * rcritv && (x4_h[i].w > 0.0 || x4_h[j].w > 0.0)){
					
					int Ni = Encpairs2_h[i * P.NencMax].x++;
					int Nj = 0;
					if(Ni >= P.NencMax){
						EncFlag_m[0] = max(EncFlag_m[0], Ni);
					}
					else{
						Encpairs2_h[i * P.NencMax + Ni].y = j;
					}

					if(P.UseTestParticles == 2){
						Nj = Encpairs2_h[j * P.NencMax].x++;

						if(Nj >= P.NencMax){
							EncFlag_m[0] = max(EncFlag_m[0], Nj);
						}
						else{
							Encpairs2_h[j * P.NencMax + Nj].y = i;
						}
					}					
//printf("enc1 small %d %d %d\n", i, j, Ni);
//printf("enc1 %d %d %d %d\n", i, j, Ni, Nj);

					if(Ni < P.NencMax && Nj < P.NencMax){
						// j < i is always true
						int Ne = Nencpairs_h[0]++;
						Encpairs_h[Ne].x = i;
						Encpairs_h[Ne].y = j;
//printf("Precheck small %d %d %d\n", i, j, Ne);
					}

					ir3 = 0.0f;
				}
				else if(rsq == 0.0){
					ir3 = 0.0f;
				}
				
				double sj = (x4_h[j].w * ir3);
				
				a_h[i].x += r3ij.x * sj;
				a_h[i].y += r3ij.y * sj;
				a_h[i].z += r3ij.z * sj;

				if(P.UseTestParticles == 2){
					double si = (x4_h[i].w * ir3);
				
					a_h[j].x -= r3ij.x * si;
					a_h[j].y -= r3ij.y * si;
					a_h[j].z -= r3ij.z * si;
				
				}
//if(i == 10) printf("acci %d %d %g %g\n", i, j, sj * r3ij.x, b_h[k * NconstT + i].x);
//if(j == 10) printf("accj %d %d %g %g\n", j, i, -si * r3ij.x, b_h[k * NconstT + j].x);
				
			}
		}
	}
}

// Can be applied only to massive particles
// Here EE is always 0
// paralle version with OpenMP
void Data::acc4D_cpu(){
	
	for(int i = 0; i < N_h[0]; ++i){
		for(int k = 0; k < Nomp; ++k){
			b_h[k * NconstT + i] = {0.0, 0.0, 0.0};
		}
		Encpairs2_h[i * P.NencMax].x = 0;
	}
	
	#pragma omp parallel for // proc_bind(close)
	for(int ii = 0; ii < N_h[0] / 2; ++ii){
		double ir, ir3;
		double3 r3ij;

		int k = omp_get_thread_num();
		//int k = 0;
//int cpuid = sched_getcpu();
//if(ii == 0 || ii == N_h[0] / 2 - 1) printf("acc4D %d %d %d %d\n", ii, k, cpuid, 0);

		for(int jj = 0; jj < N_h[0]; ++jj){
	
			if(ii == (N_h[0] + 1) / 2 - 1 && jj >= N_h[0] / 2) break;
			
			int j = jj;
			int i = ii;

			if(jj <= ii){
				i = N_h[0] - 2 - ii;
				j = N_h[0] - 1 - jj;
			}
			
			if(x4_h[i].w >= 0.0 && x4_h[j].w >= 0.0){
				
				//printf("%d %d %d %d %d\n", k, ii, jj, i, j);
				
				r3ij.x = x4_h[j].x - x4_h[i].x;
				r3ij.y = x4_h[j].y - x4_h[i].y;
				r3ij.z = x4_h[j].z - x4_h[i].z;
				
				double rsq = (r3ij.x*r3ij.x) + (r3ij.y*r3ij.y) + (r3ij.z*r3ij.z);
				
				double rcritv = fmax(rcritv_h[i], rcritv_h[j]);
				
				ir = 1.0/sqrt(rsq);
				ir3 = ir*ir*ir;
				
				
//				if(rsq < 3.0 * rcritv * rcritv){
				if(rsq < def_pc * rcritv * rcritv && (x4_h[i].w > 0.0 || x4_h[j].w > 0.0)){
					
					int Ni, Nj;
					#pragma omp atomic capture
					Ni = Encpairs2_h[i * P.NencMax].x++;
					#pragma omp atomic capture
					Nj = Encpairs2_h[j * P.NencMax].x++;

//printf("enc1 %d | %d %d %d %d\n", k, i, j, Ni, Nj);
					if(Ni >= P.NencMax){
						EncFlag_m[0] = max(EncFlag_m[0], Ni);
					}
					else{
						Encpairs2_h[i * P.NencMax + Ni].y = j;
					}
	
					if(Nj >= P.NencMax){
						EncFlag_m[0] = max(EncFlag_m[0], Nj);
					}
					else{
						Encpairs2_h[j * P.NencMax + Nj].y = i;
					}
					
					if(Ni < P.NencMax && Nj < P.NencMax){
						// i < j is always true
						int Ne;
						#pragma omp atomic capture
						Ne = Nencpairs_h[0]++;

						Encpairs_h[Ne].x = i;
						Encpairs_h[Ne].y = j;
//printf("Precheck %d | %d %d %d\n", k, i, j, Ne);
					}
					ir3 = 0.0;
				}
				else if(rsq == 0.0){
					ir3 = 0.0;
				}
				
				double si = (x4_h[i].w * ir3);
				double sj = (x4_h[j].w * ir3);
				
				int ki = k * NconstT + i; 
				int kj = k * NconstT + j; 
				
				b_h[ki].x += r3ij.x * sj;
				b_h[ki].y += r3ij.y * sj;
				b_h[ki].z += r3ij.z * sj;
				
				b_h[kj].x -= r3ij.x * si;
				b_h[kj].y -= r3ij.y * si;
				b_h[kj].z -= r3ij.z * si;
				
//if(i == 10) printf("acci %d %d %g %g\n", i, j, sj * r3ij.x, b_h[k * NconstT + i].x);
//if(j == 10) printf("accj %d %d %g %g\n", j, i, -si * r3ij.x, b_h[k * NconstT + j].x);
				
			}
		}
	}
	
	for(int i = 0; i < N_h[0]; ++i){
		double3 a = {0.0, 0.0, 0.0};
		for(int k = 0; k < Nomp; ++k){
			a.x += b_h[k * NconstT + i].x;
			a.y += b_h[k * NconstT + i].y;
			a.z += b_h[k * NconstT + i].z;
//if(i == 10) printf("%d %g %g\n", k, b_h[k * NconstT + 10].x, a.x);
			
		}
		
		a_h[i].x = a.x;
		a_h[i].y = a.y;
		a_h[i].z = a.z;
	}
	
}

// Is applied only to test particles
// parallel version with OpenMP
void Data::acc4Dsmall_cpu(){
	
	for(int i = N_h[0]; i < N_h[0] + Nsmall_h[0]; ++i){
		for(int k = 0; k < Nomp; ++k){
			b_h[k * NconstT + i] = {0.0, 0.0, 0.0};
		}
		Encpairs2_h[i * P.NencMax].x = 0;
	}
	
	#pragma omp parallel for // proc_bind(close)
	for(int i = N_h[0]; i < N_h[0] + Nsmall_h[0]; ++i){
		double ir, ir3;
		double3 r3ij;

		int k = omp_get_thread_num();
		//int k = 0;

//int cpuid = sched_getcpu();
//if(i == N_h[0] || i == N_h[0] + Nsmall_h[0] - 1) printf("acc4Dsmall %d %d %d %d\n", i, k, cpuid, 1);


		for(int j = 0; j < N_h[0]; ++j){
	
			if(x4_h[i].w >= 0.0 && x4_h[j].w >= 0.0){
				
				//printf("%d %d %d %d %d\n", k, ii, jj, i, j);
				
				r3ij.x = x4_h[j].x - x4_h[i].x;
				r3ij.y = x4_h[j].y - x4_h[i].y;
				r3ij.z = x4_h[j].z - x4_h[i].z;
				
				double rsq = (r3ij.x*r3ij.x) + (r3ij.y*r3ij.y) + (r3ij.z*r3ij.z);
				
				double rcritv = fmax(rcritv_h[i], rcritv_h[j]);
				
				ir = 1.0/sqrt(rsq);
				ir3 = ir*ir*ir;
				
				
//				if(rsq < 3.0 * rcritv * rcritv){
				if(rsq < def_pc * rcritv * rcritv && (x4_h[i].w > 0.0 || x4_h[j].w > 0.0)){
					
					int Ni = 0;
					int Nj = 0;
					#pragma omp atomic capture
					Ni = Encpairs2_h[i * P.NencMax].x++;

//printf("enc1 %d | %d %d %d %d\n", k, i, j, Ni, Nj);
					if(Ni >= P.NencMax){
						EncFlag_m[0] = max(EncFlag_m[0], Ni);
					}
					else{
						Encpairs2_h[i * P.NencMax + Ni].y = j;
					}
	
					if(P.UseTestParticles == 2){
						#pragma omp atomic capture
						Nj = Encpairs2_h[j * P.NencMax].x++;

						if(Nj >= P.NencMax){
							EncFlag_m[0] = max(EncFlag_m[0], Nj);
						}
						else{
							Encpairs2_h[j * P.NencMax + Nj].y = i;
						}
					}	
				
					if(Ni < P.NencMax && Nj < P.NencMax){
						// i < j is always true
						int Ne;
						#pragma omp atomic capture
						Ne = Nencpairs_h[0]++;

						Encpairs_h[Ne].x = i;
						Encpairs_h[Ne].y = j;
//printf("Precheck small %d | %d %d %d\n", k, i, j, Ne);
					}
					ir3 = 0.0;
				}
				else if(rsq == 0.0){
					ir3 = 0.0;
				}
				
				double sj = (x4_h[j].w * ir3);
				
				int ki = k * NconstT + i; 
				
				b_h[ki].x += r3ij.x * sj;
				b_h[ki].y += r3ij.y * sj;
				b_h[ki].z += r3ij.z * sj;

				if(P.UseTestParticles == 2){
					double si = (x4_h[i].w * ir3);
					int kj = k * NconstT + j; 
				
					b_h[kj].x -= r3ij.x * si;
					b_h[kj].y -= r3ij.y * si;
					b_h[kj].z -= r3ij.z * si;
				}
				
//if(i == 10) printf("acci %d %d %g %g\n", i, j, sj * r3ij.x, b_h[k * NconstT + i].x);
//if(j == 10) printf("accj %d %d %g %g\n", j, i, -si * r3ij.x, b_h[k * NconstT + j].x);
				
			}
		}
	}
	
	for(int i = 0; i < N_h[0]; ++i){
		double3 a = {0.0, 0.0, 0.0};
		for(int k = 0; k < Nomp; ++k){
			a.x += b_h[k * NconstT + i].x;
			a.y += b_h[k * NconstT + i].y;
			a.z += b_h[k * NconstT + i].z;
//if(i == 10) printf("%d %g %g\n", k, b_h[k * NconstT + 10].x, a.x);
			
		}
		
		a_h[i].x = a.x;
		a_h[i].y = a.y;
		a_h[i].z = a.z;
	}
	
}

//float version
void Data::acc4Df_cpu(){
	
	for(int i = 0; i < N_h[0]; ++i){
		for(int k = 0; k < Nomp; ++k){
			b_h[k * NconstT + i] = {0.0, 0.0, 0.0};
		}
		Encpairs2_h[i * P.NencMax].x = 0;
	}
	
	#pragma omp parallel for
	for(int ii = 0; ii < N_h[0] / 2; ++ii){
		float ir, ir3;
		float3 r3ij;

		int k = omp_get_thread_num();
		//int k = 0;

		for(int jj = 0; jj < N_h[0]; ++jj){
	
			if(ii == (N_h[0] + 1) / 2 - 1 && jj >= N_h[0] / 2) break;
			
			int j = jj;
			int i = ii;

			if(jj <= ii){
				i = N_h[0] - 2 - ii;
				j = N_h[0] - 1 - jj;
			}
			
			if(x4_h[i].w >= 0.0 && x4_h[j].w >= 0.0){
				
				//printf("%d %d %d %d %d\n", k, ii, jj, i, j);
				
				r3ij.x = x4_h[j].x - x4_h[i].x;
				r3ij.y = x4_h[j].y - x4_h[i].y;
				r3ij.z = x4_h[j].z - x4_h[i].z;
				
				float rsq = (r3ij.x*r3ij.x) + (r3ij.y*r3ij.y) + (r3ij.z*r3ij.z);
				
				float rcritv = fmax(rcritv_h[i], rcritv_h[j]);
				
				ir = 1.0f/sqrt(rsq);
				ir3 = ir*ir*ir;
				
				
//				if(rsq < 3.0 * rcritv * rcritv){
				if(rsq < def_pc * rcritv * rcritv && (x4_h[i].w > 0.0 || x4_h[j].w > 0.0)){
					
					int Ni, Nj;
					#pragma omp atomic capture
					Ni = Encpairs2_h[i * P.NencMax].x++;
					#pragma omp atomic capture
					Nj = Encpairs2_h[j * P.NencMax].x++;

//printf("enc1 %d | %d %d %d %d\n", k, i, j, Ni, Nj);
					if(Ni >= P.NencMax){
						EncFlag_m[0] = max(EncFlag_m[0], Ni);
					}
					else{
						Encpairs2_h[i * P.NencMax + Ni].y = j;
					}
	
					if(Nj >= P.NencMax){
						EncFlag_m[0] = max(EncFlag_m[0], Nj);
					}
					else{
						Encpairs2_h[j * P.NencMax + Nj].y = i;
					}
					
					if(Ni < P.NencMax && Nj < P.NencMax){
						// i < j is always true
						int Ne;
						#pragma omp atomic capture
						Ne = Nencpairs_h[0]++;

						Encpairs_h[Ne].x = i;
						Encpairs_h[Ne].y = j;
//printf("Precheck %d | %d %d %d\n", k, i, j, Ne);
					}
					ir3 = 0.0f;
				}
				else if(rsq == 0.0f){
					ir3 = 0.0f;
				}
				
				float si = (x4_h[i].w * ir3);
				float sj = (x4_h[j].w * ir3);
				
				int ki = k * NconstT + i; 
				int kj = k * NconstT + j; 
				
				b_h[ki].x += r3ij.x * sj;
				b_h[ki].y += r3ij.y * sj;
				b_h[ki].z += r3ij.z * sj;
				
				b_h[kj].x -= r3ij.x * si;
				b_h[kj].y -= r3ij.y * si;
				b_h[kj].z -= r3ij.z * si;
				
//if(i == 10) printf("acci %d %d %g %g\n", i, j, sj * r3ij.x, b_h[k * NconstT + i].x);
//if(j == 10) printf("accj %d %d %g %g\n", j, i, -si * r3ij.x, b_h[k * NconstT + j].x);
				
			}
		}
	}
	
	for(int i = 0; i < N_h[0]; ++i){
		double3 a = {0.0, 0.0, 0.0};
		for(int k = 0; k < Nomp; ++k){
			a.x += b_h[k * NconstT + i].x;
			a.y += b_h[k * NconstT + i].y;
			a.z += b_h[k * NconstT + i].z;
//if(i == 10) printf("%d %g %g\n", k, b_h[k * NconstT + 10].x, a.x);
			
		}
		
		a_h[i].x = a.x;
		a_h[i].y = a.y;
		a_h[i].z = a.z;
	}
	
}

// Is applied only to test particles
// parallel version with OpenMP
void Data::acc4Dfsmall_cpu(){
	
	for(int i = N_h[0]; i < N_h[0] + Nsmall_h[0]; ++i){
		for(int k = 0; k < Nomp; ++k){
			b_h[k * NconstT + i] = {0.0, 0.0, 0.0};
		}
		Encpairs2_h[i * P.NencMax].x = 0;
	}
	
	#pragma omp parallel for
	for(int i = N_h[0]; i < N_h[0] + Nsmall_h[0]; ++i){
		float ir, ir3;
		float3 r3ij;

		int k = omp_get_thread_num();
		//int k = 0;

		for(int j = 0; j < N_h[0]; ++j){
	
			if(x4_h[i].w >= 0.0 && x4_h[j].w >= 0.0){
				
				//printf("%d %d %d %d %d\n", k, ii, jj, i, j);
				
				r3ij.x = x4_h[j].x - x4_h[i].x;
				r3ij.y = x4_h[j].y - x4_h[i].y;
				r3ij.z = x4_h[j].z - x4_h[i].z;
				
				float rsq = (r3ij.x*r3ij.x) + (r3ij.y*r3ij.y) + (r3ij.z*r3ij.z);
				
				float rcritv = fmax(rcritv_h[i], rcritv_h[j]);
				
				ir = 1.0f/sqrt(rsq);
				ir3 = ir*ir*ir;
				
				
//				if(rsq < 3.0 * rcritv * rcritv){
				if(rsq < def_pc * rcritv * rcritv && (x4_h[i].w > 0.0 || x4_h[j].w > 0.0)){
					
					int Ni = 0;
					int Nj = 0;
					#pragma omp atomic capture
					Ni = Encpairs2_h[i * P.NencMax].x++;

//printf("enc1 %d | %d %d %d %d\n", k, i, j, Ni, Nj);
					if(Ni >= P.NencMax){
						EncFlag_m[0] = max(EncFlag_m[0], Ni);
					}
					else{
						Encpairs2_h[i * P.NencMax + Ni].y = j;
					}
	
					if(P.UseTestParticles == 2){
						#pragma omp atomic capture
						Nj = Encpairs2_h[j * P.NencMax].x++;

						if(Nj >= P.NencMax){
							EncFlag_m[0] = max(EncFlag_m[0], Nj);
						}
						else{
							Encpairs2_h[j * P.NencMax + Nj].y = i;
						}
					}	
				
					if(Ni < P.NencMax && Nj < P.NencMax){
						// i < j is always true
						int Ne;
						#pragma omp atomic capture
						Ne = Nencpairs_h[0]++;

						Encpairs_h[Ne].x = i;
						Encpairs_h[Ne].y = j;
//printf("Precheck %d | %d %d %d\n", k, i, j, Ne);
					}
					ir3 = 0.0f;
				}
				else if(rsq == 0.0f){
					ir3 = 0.0f;
				}
				
				float sj = (x4_h[j].w * ir3);
				
				int ki = k * NconstT + i; 
				
				b_h[ki].x += r3ij.x * sj;
				b_h[ki].y += r3ij.y * sj;
				b_h[ki].z += r3ij.z * sj;

				if(P.UseTestParticles == 2){
					float si = (x4_h[i].w * ir3);
					int kj = k * NconstT + j; 
				
					b_h[kj].x -= r3ij.x * si;
					b_h[kj].y -= r3ij.y * si;
					b_h[kj].z -= r3ij.z * si;
				}
				
//if(i == 10) printf("acci %d %d %g %g\n", i, j, sj * r3ij.x, b_h[k * NconstT + i].x);
//if(j == 10) printf("accj %d %d %g %g\n", j, i, -si * r3ij.x, b_h[k * NconstT + j].x);
				
			}
		}
	}
	
	for(int i = 0; i < N_h[0]; ++i){
		double3 a = {0.0, 0.0, 0.0};
		for(int k = 0; k < Nomp; ++k){
			a.x += b_h[k * NconstT + i].x;
			a.y += b_h[k * NconstT + i].y;
			a.z += b_h[k * NconstT + i].z;
//if(i == 10) printf("%d %g %g\n", k, b_h[k * NconstT + 10].x, a.x);
			
		}
		
		a_h[i].x = a.x;
		a_h[i].y = a.y;
		a_h[i].z = a.z;
	}
	
}
#endif


//******************************************************
// This function calculates the force between body i and j
// it must be called n^2/2 times

//Author: Simon Grimm, Joachim Stadel
// January 2015
//********************************************************




//******************************************************
// This kernel perfomes a Kick operation on the triangle part
// of the interaction matrix
// the two indexes I and II must come from a driver routine

//p: number of threads per block, it is set in the driver routine
//nb:number of threadsblock, it is set in the driver routine

//Author: Simon Grimm, Joachim Stadel
// January 2015
//********************************************************


//******************************************************
// This kernel perfomes a Kick operation on blocks on the diagonal part
// of the interaction matrix in single precision

//p: number of threads per block, it is set in the driver routine

//Author: Simon Grimm, Joachim Stadel
// January 2015
//********************************************************

//******************************************************
// This kernel perfomes a Kick operation on the lower left square part
// of the interaction matrix
// the index I must come from a driver routine

//p: number of threads per block, it is set in the driver routine
//nb:number of threadsblock, it is set in the driver routine

//Author: Simon Grimm, Joachim Stadel
// January 2015
//********************************************************




//******************************************************
// this function is a driver for the Kick kernels
// it splits thes N * N matrix into smaller blocks of lenght p * p
// this blocks are devided into 3 sets:
// set 1: blocks on the diagonal
// set 2: upper left triangle and lowet right triangle
// set 3: lower left square

// p sets the size of the blocks and the number of threads per block

//Author: Simon Grimm, Joachim Stadel
// January 2015
//********************************************************




#endif
