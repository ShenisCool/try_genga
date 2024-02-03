#ifndef ENCOUNTER_H
#define ENCOUNTER_H
#include "Orbit2CPU.h"

#if def_G3 ==1
#include "Encounter3G3.h"
#endif


void setNencpairs_cpu(int *Nencpairs_h, const int N){

	int id = 0 * 1 + 0;
	for(id = 0 * 1 + 0; id < N; ++id){
		Nencpairs_h[id] = 0;
	}
}


// **************************************
//This function estimates the minimal separation of two bodies
//during a time step, using a third order interpolation. 
//
//The interpolation scheme is based on the mercury code from Chambers.
//
//If the minimal separation is less than the critical radius, a
//close encounter is reported.
//
//
// Authors: Simon Grimm
// April 2016
//
// ****************************************
int encounter(const double4 x4i, const double4 v4i, const double4 x4oldi, const double4 v4oldi, const double4 x4j, const double4 v4j, const double4 x4oldj, const double4 v4oldj, const double rcriti, const double rcritj, const double rcritvi, const double rcritvj, const double dt, const int i, const int j, double &time, const double MinMass){

//printf("E0 %d %d %.20g %.20g %.20g %.20g %.20g %.20g | %.20g %.20g %.20g %.20g %.20g %.20g | m %.20g %.20g\n", i ,j, x4oldi.x, x4oldi.y, x4oldi.z, v4oldi.x, v4oldi.y, v4oldi.z, x4oldj.x, x4oldj.y, x4oldj.z, v4oldj.x, v4oldj.y, v4oldj.z, x4oldi.w, x4oldj.w);
//printf("E1 %d %d %.20g %.20g %.20g %.20g %.20g %.20g | %.20g %.20g %.20g %.20g %.20g %.20g | m %.20g %.20g\n", i ,j, x4i.x, x4i.y, x4i.z, v4i.x, v4i.y, v4i.z, x4j.x, x4j.y, x4j.z, v4j.x, v4j.y, v4j.z, x4i.w, x4j.w);
	int Enc = 0;
	if(i != j && (x4i.w > MinMass || x4j.w > MinMass) && x4i.w >= 0.0 && x4j.w >= 0.0){
		double d0, d1, dd0, dd1;
		double4 r1, r0;
		double4 rd0, rd1;
		double a,b,c,cc;
		double w,q;
		double t1,t2,t12,t22,tt1,tt2,tt12,tt22;
		double delta1, delta2;
		double delta;
		double sgnb;
		double rcrit;
		double rcritv;
		double f;
	
		rcrit = fmax(rcriti, rcritj);
		rcritv = fmax(rcritvi, rcritvj);
		f = def_cef;

		r1.x = x4j.x - x4i.x;
		r1.y = x4j.y - x4i.y;
		r1.z = x4j.z - x4i.z;
		d1 = r1.x*r1.x + r1.y*r1.y+ r1.z*r1.z;

		r0.x = x4oldj.x - x4oldi.x;
		r0.y = x4oldj.y - x4oldi.y;
		r0.z = x4oldj.z - x4oldi.z;
		d0 = r0.x*r0.x + r0.y*r0.y+ r0.z*r0.z;
			
		rd0.x = v4oldj.x - v4oldi.x;
		rd0.y = v4oldj.y - v4oldi.y;
		rd0.z = v4oldj.z - v4oldi.z;

		rd1.x = v4j.x - v4i.x;
		rd1.y = v4j.y - v4i.y;
		rd1.z = v4j.z - v4i.z;

		dd0 = (r0.x*rd0.x + r0.y*rd0.y+ r0.z*rd0.z) * 2.0;
		dd1 = (r1.x*rd1.x + r1.y*rd1.y+ r1.z*rd1.z) * 2.0;
		t1 = 6.0 *(d0-d1); 
		a = t1 + 3.0*dt*(dd0+dd1);
		b = -t1 - 2.0*dt*(2.0*dd0+ dd1);
		c = dt*dd0;
		cc = dt*dd1;

		if(b < 0){
			sgnb = -1.0;
		}
		else sgnb = 1.0;
		t1 = 0.0;
		t2 = 0.0;

		w = b*b - 4.0*a*c;
		if(w < 0.0) w = 0.0;
		if( b != 0){
			q = -0.5 * (b + sgnb * sqrt(w));
			if(q != 0){
				if( a != 0){
					t1 = q/a;
					t2 = c/q;
				}
				else{
					t1 = -c/b;
					t2 = t1;
				}
			}	
		}
		else{
			if( a != 0){
				t1 = sqrt(-c/a);
				t2 = -t1;
			}
		}

//printf("dt %d %d %g %g %g\n", i, j, t1, t2, dt);

		if(0 <= t1 && t1 <= 1){
			t12 = t1*t1;
			tt1 = 1.0-t1;
			tt12 = tt1*tt1;
			delta1 = tt12*(1.0 + 2.0*t1)*d0 + t12*(3.0 - 2.0*t1)*d1 + t1*tt12*c - t12*tt1*cc;
		}
		else delta1 = 100.0;
		if(0 <= t2 && t2 <= 1){
			t22 = t2*t2;
			tt2 = 1.0-t2;
			tt22 = tt2*tt2;
			delta2 = tt22*(1.0 + 2.0*t2)*d0 + t22*(3.0 - 2.0*t2)*d1 + t2*tt22*c - t22*tt2*cc;
		}
		else delta2 = 100.0;

		delta = fmin(delta1,delta2);
		if(delta < 0) delta = 0.0;
		
		delta = fmin(delta, d1);
		delta = fmin(delta, d0);
//printf("EE %d %d %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %g %g %g %g %g %g\n", i, j, time, x4i.w, x4j.w, x4i.x, x4i.y, x4i.z, x4j.x, x4j.y, x4j.z, delta, rcritv*rcritv, d0, d1, delta1, delta2, t1, t2);

		if(delta < f * rcritv*rcritv){
			Enc = 2;
//printf("EE %d %d %g %g %.40g %.40g %.40g %.40g %g %g\n", i, j, x4i.w, x4j.w, x4i.x, x4i.y, v4i.z, v4j.x, v4j.y, v4j.z);
		}
		else Enc = 0;

		if(delta < rcrit*rcrit){
			Enc = 1;
		}

		return Enc;
	}
	else return 0;
}
int encounterb(const double4 x4i, const double4 v4i, const double4 x4oldi, const double4 v4oldi, const double4 x4j, const double4 v4j, const double4 x4oldj, const double4 v4oldj, const double rcriti, const double rcritj, const double rcritvi, const double rcritvj, const double dt, const int i, const int j, double &Ki, double &Kj, double &Kiold, double &Kjold, double &time, const double MinMass){

//printf("E %d %d %.20g %.20g %.20g %.20g %.20g %.20g\n", i , j, x4oldi.x, x4oldi.y, x4oldi.z, v4oldi.x, v4oldi.y, v4oldi.z);

	int Enc = 0;
	if(i < j && (x4i.w > MinMass || x4j.w > MinMass) && x4i.w >= 0.0 && x4j.w >= 0.0){
		double d0, d1, dd0, dd1;
		double4 r1, r0;
		double4 rd0, rd1;
		double a,b,c,cc;
		double w,q;
		double t1,t2,t12,t22,tt1,tt2,tt12,tt22;
		double delta1, delta2;
		double delta;
		double sgnb;
		double rcrit;
		double rcritv;
		double f;
	
		rcrit = fmax(rcriti, rcritj);
		rcritv = fmax(rcritvi, rcritvj);
		f = def_cef;

		r1.x = x4j.x - x4i.x;
		r1.y = x4j.y - x4i.y;
		r1.z = x4j.z - x4i.z;
		d1 = r1.x*r1.x + r1.y*r1.y+ r1.z*r1.z;

		r0.x = x4oldj.x - x4oldi.x;
		r0.y = x4oldj.y - x4oldi.y;
		r0.z = x4oldj.z - x4oldi.z;
		d0 = r0.x*r0.x + r0.y*r0.y+ r0.z*r0.z;
			
		rd0.x = v4oldj.x - v4oldi.x;
		rd0.y = v4oldj.y - v4oldi.y;
		rd0.z = v4oldj.z - v4oldi.z;

		rd1.x = v4j.x - v4i.x;
		rd1.y = v4j.y - v4i.y;
		rd1.z = v4j.z - v4i.z;

		dd0 = (r0.x*rd0.x + r0.y*rd0.y+ r0.z*rd0.z) * 2.0;
		dd1 = (r1.x*rd1.x + r1.y*rd1.y+ r1.z*rd1.z) * 2.0;
		t1 = 6.0 *(d0-d1); 
		a = t1 + 3.0*dt*(dd0+dd1);
		b = -t1 - 2.0*dt*(2.0*dd0+ dd1);
		c = dt*dd0;
		cc = dt*dd1;

		if(b < 0){
			sgnb = -1.0;
		}
		else sgnb = 1.0;
		t1 = 0.0;
		t2 = 0.0;

		w = b*b - 4.0*a*c;
		if(w < 0.0) w = 0.0;
		if( b != 0){
			q = -0.5 * (b + sgnb * sqrt(w));
			if(q != 0){
				if( a != 0){
					t1 = q/a;
					t2 = c/q;
				}
				else{
					t1 = -c/b;
					t2 = t1;
				}
			}	
		}
		else{
			if( a != 0){
				t1 = sqrt(-c/a);
				t2 = -t1;
			}
		}

		if(0 <= t1 && t1 <= 1){
			t12 = t1*t1;
			tt1 = 1.0-t1;
			tt12 = tt1*tt1;
			delta1 = tt12*(1.0 + 2.0*t1)*d0 + t12*(3.0 - 2.0*t1)*d1 + t1*tt12*c - t12*tt1*cc;
		}
		else delta1 = 100.0;
		if(0 <= t2 && t2 <= 1){
			t22 = t2*t2;
			tt2 = 1.0-t2;
			tt22 = tt2*tt2;
			delta2 = tt22*(1.0 + 2.0*t2)*d0 + t22*(3.0 - 2.0*t2)*d1 + t2*tt22*c - t22*tt2*cc;
		}
		else delta2 = 100.0;

		delta = fmin(delta1,delta2);
		if(delta < 0) delta = 0.0;
		
		delta = fmin(delta, d1);
		delta = fmin(delta, d0);

//printf("d %d %d %.20g %.20g\n", i, j, delta, rcritv);
	
		Kiold = Ki;
		Kjold = Kj;
		Ki = 1.0;
		Kj = 1.0;


		if(delta < f * rcritv*rcritv || Kiold < 1.0){
			Enc = 2;
//printf("EE %d %d %g %g %.40g %.40g %.40g %.40g\n", i, j, x4i.w, x4j.w, x4i.x, x4j.x, v4i.x, v4j.x);

			if(delta <= 0.01 * rcritv*rcritv){
		 		Ki = 0.0;
		 		Kj = 0.0;
			
			}
			else{
				double y = (sqrt(delta) - 0.1 * rcritv)/(0.9*rcritv);
				double yy = y * y;
				Ki = yy / (2.0*yy - 2.0*y + 1.0);
				Kj = yy / (2.0*yy - 2.0*y + 1.0);
			}

		}
		else Enc = 0;
		if(delta < rcrit*rcrit){
			Enc = 1;
		}

//printf("Enc %d %d %g %g\n", i, j, Ki, Kiold);

		return Enc;
	}
	else return 0;
}


// This function returns delta, the square of the minimal distanz
// It sets colt, the collision time  0 < colt < 1.
// It sets enct, the time of smallest distance. 0 < enct < 1
double encounter1(const double4 x4i, const double4 v4i, const double4 x4oldi, const double4 v4oldi, const double4 x4j, const double4 v4j, const double4 x4oldj, const double4 v4oldj, const double rcrit, const double dt, const int i, const int j, double &enct, double &colt, const double MinMass, const int noColl){

//printf("E1o %d %d %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g\n", i, j, x4oldi.w, v4oldi.w, x4oldi.x, x4oldi.y, x4oldi.z, x4oldj.w, v4oldj.w, x4oldj.x, x4oldj.y, x4oldj.z);
	if(i != j && (x4i.w > MinMass || x4j.w > MinMass) && x4i.w >= 0.0 && x4j.w >= 0.0){

		double d0, d1, dd0, dd1;
		double3 r;
		double3 rd;

		r.x = x4j.x - x4i.x;
		r.y = x4j.y - x4i.y;
		r.z = x4j.z - x4i.z;
		d1 = r.x*r.x + r.y*r.y+ r.z*r.z;

		rd.x = v4j.x - v4i.x;
		rd.y = v4j.y - v4i.y;
		rd.z = v4j.z - v4i.z;

		dd1 = (r.x*rd.x + r.y*rd.y+ r.z*rd.z) * 2.0;

		r.x = x4oldj.x - x4oldi.x;
		r.y = x4oldj.y - x4oldi.y;
		r.z = x4oldj.z - x4oldi.z;
		d0 = r.x*r.x + r.y*r.y+ r.z*r.z;
			
		rd.x = v4oldj.x - v4oldi.x;
		rd.y = v4oldj.y - v4oldi.y;
		rd.z = v4oldj.z - v4oldi.z;

		dd0 = (r.x*rd.x + r.y*rd.y+ r.z*rd.z) * 2.0;

		double t1, t2;
		t1 = 6.0 *(d0-d1); 
		double a = t1 + 3.0 * dt * (dd0 + dd1);
		double b = -t1 - 2.0 * dt * (2.0 * dd0 + dd1);
		double c = dt*dd0;

		double sgnb = 1.0;
		if(b < 0){
			sgnb = -1.0;
		}
		t1 = 0.0;
		t2 = 0.0;

		double w = b*b - 4.0*a*c;
		if(w < 0.0) w = 0.0;
		if( b != 0){
			double q = -0.5 * (b + sgnb * sqrt(w));
			if(q != 0){
				if( a != 0){
					t1 = q/a;
					t2 = c/q;
				}
				else{
					t1 = -c/b;
					t2 = t1;
				}
			}	
		}
		else{
			if( a != 0){
				t1 = sqrt(-c/a);
				t2 = -t1;
			}
		}
//if(i == 12888 && j == 11191) printf("d0d1  %d %d %g %g | %.12g %.12g %.12g | noColl: %d\n", i, j, t1, t2, rcrit, sqrt(d0), sqrt(d1), noColl);
//if(i == 4969 && j == 530) printf("d0d1  %d %d %g %g | %.12g %.12g %.12g | noColl: %d\n", i, j, t1, t2, rcrit, sqrt(d0), sqrt(d1), noColl);
		double delta = 100.0;
		if(0 <= t1 && t1 <= 1){
			double t12 = t1*t1;
			double tt1 = 1.0-t1;
			double tt12 = tt1*tt1;
			double delta1 = tt12*(1.0 + 2.0*t1)*d0 + t12*(3.0 - 2.0*t1)*d1 + t1*tt12*dt*dd0 - t12*tt1*dt*dd1;
			delta = fmin(delta, delta1);
			enct = t1;
		}
		if(0 <= t2 && t2 <= 1){
			double t22 = t2*t2;
			double tt2 = 1.0-t2;
			double tt22 = tt2*tt2;
			double delta2 = tt22*(1.0 + 2.0*t2)*d0 + t22*(3.0 - 2.0*t2)*d1 + t2*tt22*dt*dd0 - t22*tt2*dt*dd1;
			delta = fmin(delta, delta2);
			enct = t2;
		}
		if(delta < 0) delta = 0.0;
	
		delta = fmin(delta, d1);
		delta = fmin(delta, d0);

//if(enct >= 0.0 && enct <= 1.0) printf("dt %d %d %g %g %g %g %g\n", i, j, sqrt(delta), enct, t1, t2, rcrit);
		double rcritsq = rcrit * rcrit;

		if(delta < rcritsq){
//printf("EEa %d %d %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %g %g %g %g %g %g\n", i, j, x4i.w, x4j.w, x4i.x, x4i.y, x4i.z, x4j.x, x4j.y, x4j.z, delta, rcritsq, d0, d1, delta, t1, t2, MinMass);
			if(d0 == d1){
				colt = 0.0;
			}
			else{
				double coltime = (rcritsq - d0) / (d1 - d0);
				
				if(noColl == 1){
					//only report entering collisions, not leaving collisions
					if(d1 <= rcritsq && d0 > rcritsq){
						//linear interpolation of the collision time
						colt = coltime;
					}
		
					//find collision in between of time steps	
					if(d0 >= rcritsq && d1 >= rcritsq){
						colt = fmin(t1, t2);
					}

					if(d0 < 0.95 * rcritsq && d1 < 0.95 * rcritsq){
//printf("%d %d are already overlapping %g %g %g\n", i, j, sqrt(d0), sqrt(d1), rcrit);
						colt = 200.0;
						return 100.0;
					}

				}
				else if(noColl == -1){
					//only report leaving collisions, not entering collisions
					if(d0 <= rcritsq && d1 > rcritsq){
						//linear interpolation of the collision time
						colt = coltime;
					}
				}
				else{
					if((d0 >= rcritsq && d1 < rcritsq) || (d1 >= rcritsq && d0 < rcritsq)){
						//linear interpolation of the collision time
						colt = coltime;
					}


					//find collision in between of time steps	
					if(d0 >= rcritsq && d1 >= rcritsq){
						if(t1 < 0.0) t1 = t2;
						if(t2 < 0.0) t2 = t1;
						if(t1 >= 0.0){
							colt = fmin(t1, t2);
						}
					}


				}
//printf("colt  %d %d %.12g %.12g %.12g | noColl: %d\n", i, j, coltime, colt, enct, noColl);
			}
		}
		return delta;

	}
	else return 100.0;
}

// **************************************
// For the multi simulation mode
// This reads all encounter pairs from the prechecker, and calls the encounter function
// to detect close encounter pairs.
// All close encounter pairs are stored in the array Encpairs2_h. 
// The number of close encounter pairs is stored in Nencpairs2_h.
// Creates a list of simulations containing close encounters
//
// Authors: Simon Grimm
// July  2016
//
// ****************************************

// **************************************
// Does not creates a list of simulations containing close encounters
// All simulations call the group kernel
// That is needed because in that way the full grouping algorithm can be applied, not just M1 grouping
//
// Authors: Simon Grimm
// March  2023
//
// ****************************************

// **************************************
//This kernels reads all encounter pairs from the prechecker, and calls the encounter function
//to detect close encounter pairs.
//All close encounter pairs are stored in the array Encpairs2_h. 
//The number of close encounter pairs is stored in Nencpairs2_h.
//
//Authors: Simon Grimm
//April 2019
//
// ****************************************
void encounter_cpu(double4 *x4_h, double4 *v4_h, double4 *xold_h, double4 *vold_h, double *rcrit_h, double *rcritv_h, const double dt, int Nencpairs, int *Nencpairs_h, int2 *Encpairs_h, int *Nencpairs2_h, int2 *Encpairs2_h, unsigned int *enccount_h, const int si, const int NB, double time, const int StopAtEncounter, int *Ncoll_m, const double MinMass){

	int id = 0 * 1 + 0;

	#pragma omp parallel for
	for(id = 0 * 1 + 0; id < Nencpairs; ++id){
		int ii = Encpairs_h[id].x;
		int jj = Encpairs_h[id].y;
//printf("%d %d %d\n", ii, jj, id);
		int enccount = 0;	

#if def_G3 == 0
		enccount = encounter(x4_h[ii], v4_h[ii], xold_h[ii], vold_h[ii], x4_h[jj], v4_h[jj], xold_h[jj], vold_h[jj], rcrit_h[ii], rcrit_h[jj], rcritv_h[ii], rcritv_h[jj], dt, ii, jj , time, MinMass);
#elif def_G3 == 1
		enccount = encounterb(x4_h[ii], v4_h[ii], xold_h[ii], vold_h[ii], x4_h[jj], v4_h[jj], xold_h[jj], vold_h[jj], rcrit_h[ii], rcrit_h[jj], rcritv_h[ii], rcritv_h[jj], dt, ii, jj , K_h[ii * NB + jj], K_h[jj * NB + ii], Kold_h[ii * NB + jj], Kold_h[jj * NB + ii], time, MinMass);
#else
//change here ii and jj to index[ii], index[jj]
		enccount = encounterG3(x4_h[ii], v4_h[ii], xold_h[ii], vold_h[ii], x4G3_d[ii], v4G3_d[ii], x4_h[jj], v4_h[jj], xold_h[jj], vold_h[jj], x4G3_d[jj], v4G3_d[jj], rcrit_h[ii], ii, jj, rcrit_h[jj], rcritv_h[ii], rcritv_h[jj], dt, ii, jj , Encpairs2_h, *Nencpairs2_h, 0, K_h[ii * NB + jj], K_h[jj * NB + ii], Kold_h[ii * NB + jj], Kold_h[jj * NB + ii], time, MinMass);
#endif
		if(enccount > 0){
#if def_CPU == 0
			int Ne = atomicAdd(Nencpairs2_h, 1);
#else
			int Ne;
			#pragma omp atomic capture
			Ne = Nencpairs2_h[0]++;
#endif
			if(StopAtEncounter > 0){
				if(enccount == 1){
					Ncoll_m[0] = 1;
				}
			}

			if(x4_h[ii].w >= x4_h[jj].w){
				Encpairs2_h[Ne].x = ii;
				Encpairs2_h[Ne].y = jj;
			}
			else{
				Encpairs2_h[Ne].x = jj;
				Encpairs2_h[Ne].y = ii;
			}

// *****************
//dont group test particles
/*
			if(x4_h[ii].w == 0.0){
				Encpairs2_h[Ne].x = ii;
				Encpairs2_h[Ne].y = jj;
			}
			if(x4_h[jj].w == 0.0){
				Encpairs2_h[Ne].x = jj;
				Encpairs2_h[Ne].y = ii;
			}
*/
// *****************

		}
		if(si == 0 && enccount > 0){
#if def_CPU == 0
			atomicAdd(&enccount_h[ii], 1);
			atomicAdd(&enccount_h[jj], 1);
#else
			#pragma omp atomic
			enccount_h[ii]++;
			#pragma omp atomic
			enccount_h[jj]++;
#endif
		}
		if(id == 0){
			Nencpairs_h[0] = 0;
		}
	}
}

// **************************************
// This kernels reads all encounter pairs between test particles. It calls the encounter function
// to detect close encounter pairs. Close encounters are reported in the writeencounters file.
//
// Authors: Simon Grimm
// October 2022
// ****************************************
void encounter_small_cpu(double4 *x4_h, double4 *v4_h, double4 *xold_h, double4 *vold_h, int *index_h, double4 *spin_h, const double dt, int Nencpairs2, int2 *Encpairs2_h, int *NWriteEnc_m, double *writeEnc_h, double time){

	int id = 0 * 1 + 0;

	for(id = 0 * 1 + 0; id < Nencpairs2; ++id){

		int ii = Encpairs2_h[id].x;
		int jj = Encpairs2_h[id].y;
		if(ii < jj){
			int swap = jj;
			jj = ii;
			ii = swap;
		}
//printf("%d %d %d\n", ii, jj, id);


		double delta = 1000.0;
		double enct = 100.0;
		double colt = 100.0;
		double rcrit = WriteEncountersRadius_c[0] * fmax(v4_h[ii].w, v4_h[jj].w); //writeradius

		delta = encounter1(x4_h[ii], v4_h[ii], xold_h[ii], vold_h[ii], x4_h[jj], v4_h[jj], xold_h[jj], vold_h[jj], rcrit, dt, ii, jj, enct, colt, 0.0, 0);

		if(delta < rcrit * rcrit){
			if(enct > 0.0 && enct < 1.0){
//printf("Write Enc %g %g %g %g %g %d %d\n", dt / dayUnit, rcrit, sqrt(delta), enct, colt, ii, jj);
#if def_CPU == 0
				int ne = atomicAdd(NWriteEnc_m, 1);
#else
				int ne;
				#pragma omp atomic capture
				ne = NWriteEnc_m[0]++;
#endif
				if(ne >= def_MaxWriteEnc - 1) ne = def_MaxWriteEnc - 1;
				if(colt > 1.0) colt = 1.0;
				storeEncounters(x4_h, v4_h, ii, jj, ii, jj, index_h, ne, writeEnc_h, time + (colt * dt - dt) / dayUnit, spin_h);
			}
		}
	}
}


#if def_CPU == 1
void group_cpu(int *Nenc_m, int *Nencpairs2_h, int2 *Encpairs2_h, int2 *Encpairs_h, const int NencMax, const int NT, const int N, const int SERIAL_GROUPING){


	int T_s;
	int start_s[1];

	int Ne = *Nencpairs2_h;
	

	int BN2 = NT * NT -1;
	if(NT > 46340) BN2 = 2147483647;	//prevent from overflow

	int2 *A;
	int2 *encpairs;

	int2 *B;
	int2 *B2;

	A = &Encpairs2_h[Ne];
	encpairs = Encpairs2_h;
	for(int i = 0; i < Ne; ++i){
		A[i] = encpairs[i];
	}

	B = &Encpairs_h[2 * NT];
	B2 = &Encpairs_h[3 * NT];
	for(int i = 0; i < NT; ++i){
		B[i].y = BN2;
		B2[i].y = BN2;
		Encpairs_h[i].y = 0;
	}

	T_s = 1;
	start_s[0] = 0;

	for(int i = 0; i < def_GMax; ++i){
		Nenc_m[i] = 0;
	}

	for(int i = 0; i < Ne; ++i){
		//create list of direct close encounter pairs
		volatile int ii = encpairs[i].x;
		volatile int jj = encpairs[i].y;
		volatile int Ni = 0;
		volatile int Nj = 0;
		if(jj < N){
			Ni = Encpairs_h[ii].y++;
			Encpairs_h[ii * NencMax + Ni].x = jj;
		}
		if(ii < N){
			Nj = Encpairs_h[jj].y++;
			Encpairs_h[jj * NencMax + Nj].x = ii;
		}
		//Encpairs_h[i].y contains the number of direct encounter pairs of body i
		//Encpairs_h[i * NencMax + j].x contains the indeces j of the direct encounter pairs
//printf("Encpairs %d %d %d %d\n", ii, jj, Ni, Nj);
	}
	if(SERIAL_GROUPING == 1){
		for(int i = 0; i < NT; ++i){
			int Ni = Encpairs_h[i].y;
			int stop = 0;
			while(stop == 0){
				stop = 1;
				for(int j = 0; j < Ni - 1; ++j){
					int jj = Encpairs_h[i * NencMax + j].x;
					int jjnext = Encpairs_h[i * NencMax + j + 1].x;
				
					if(jjnext < jj){
						//swap
						Encpairs_h[i * NencMax + j].x = jjnext;
						Encpairs_h[i * NencMax + j + 1].x = jj;
						stop = 0;
					}
				}
			}
			stop = 0;
		}
	}

	for(int tt = 0; tt < 100; ++tt){ 
		T_s = 0;
		for(int i = 0; i < Ne; ++i){
			int Am = min(A[i].x, A[i].y);
			B[A[i].y].y = min(B[A[i].y].y, Am);			
			B[A[i].x].y = min(B[A[i].x].y, Am);			
		}

		for(int i = 0; i < NT; ++i){
			if(B[i].y < BN2) B2[i].y = B[B[i].y].y;
		}
		for(int i = 0; i < Ne; ++i){
			A[i].x = B2[encpairs[i].x].y;
			A[i].y = B2[encpairs[i].y].y;
			if(A[i].x != A[i].y) T_s = 1;
		}
		for(int i = 0; i < NT; ++i){
			B[i].y = B2[i].y;
		}
		if(T_s == 0){
			 break;
		}

	}
	// At this point B[i] contains the smallest index of the group

	for(int i = 0; i < NT; ++i){
		B2[i].y = -1;
//printf("B %d %d\n", i, B[i].y);
	}
	// Check now for new groups and increase the total number of groups
	for(int i = 0; i < NT; ++i){
		if(B[i].y == i){
			B2[i].y = Nenc_m[0]++;
		}		
	}
	// Transform now the smallest index of the group into a consecutive group index
	for(int i = 0; i < NT; ++i){
		if(B[i].y < BN2) B[i].y = B2[B[i].y].y;
		Encpairs2_h[i].y = 0;
	}
	// At this point B[i] contains a consecutive group index

//for(int i = 0; i < NT; ++i){
//	if(B[i].y < BN2){
//printf("B %d %d %d\n", i, B[i].y, B2[i].y);
//	}		
//}

	if(SERIAL_GROUPING == 0){
		for(int i = 0; i < NT; ++i){
			if(B[i].y < BN2){
				int Ns = Encpairs2_h[B[i].y].y++;
				B2[i].y = Ns; //index in the group
				Encpairs_h[NT + i].y = B2[i].y;
			}
			// At this point Encpairs2_h.x contains now line by line the members of the groups, Encpairs2_s.y contains the sizes of the groups
		}
	}
	if(SERIAL_GROUPING == 1){
		for(int i = NT - 1; i >=0; --i){
			if(B[i].y < BN2){
				int Ns = Encpairs2_h[B[i].y].y++;
				B2[i].y = Ns;	//index in the group
				Encpairs_h[NT + i].y = B2[i].y;
			}
		}
	}
	for(int i = 0; i < Nenc_m[0]; ++i){
		if(Encpairs2_h[i].y > 0){
			int start = start_s[0];
			start_s[0] += Encpairs2_h[i].y;
			Encpairs2_h[NT + i].y = start; //starting points of te groups
//printf("start %d %d %d %d\n", i, Encpairs2_h[i].y, start_s[0], start);
		}
	}
	for(int i = 0; i < NT; ++i){
		if(B[i].y < BN2){
			int n = B2[i].y;
			int start = Encpairs2_h[NT + B[i].y].y;
			Encpairs2_h[start + n].x = i;
//printf("members %d %d %d\n", start, n, i);
		}
		// At this point Encpairs2_h.x contains now members of the groups, Encpairs2_h.y contains the sizes of the groups/
	}

	for(int i = 0; i < NT; ++i){
		int nn = Encpairs2_h[i].y;
//if(nn > 0) printf("n %d %d\n", i, nn);
		volatile int ne2 = 2;
		if(nn > 0){
			for(volatile int ii = 0; ii < def_GMax - 1; ++ii){
				if(nn <= ne2){
					int Ns = Nenc_m[ii + 1]++;
//printf("G %d %d\n", ii + 1, Nenc_m[ii + 1]);
					Encpairs2_h[ (ii+2) * NT + Ns].y = i;
					break;
				} 
				else{
					ne2 *= 2;
				}
			}
		}
	}

}
#endif

// **************************************
//This Kernel sorts all close encounter pairs into independent groups, using a 
//parallel sorting algorithm. 
//this kernel works for the following cases:
// E = 1: less than 512 bodies and less than 512 close encounter pairs
// E = 2: less than 512 bodies and more than 512 close encounter pairs
// E = 3: more than 512 bodies and less than 512 close encounter pairs
// E = 4: more than 512 bodies and more than 512 close encounter pairs
//It classifies the groups into sets of equal sizes.
//The size of group i is stored in Encpairs2_h[i].y, the elements j of the 
//group i are stored in Encpairs2_h[i * N + j].x
//In Nenc_m[0] is stored the total number of groups.
//in Nenc_m[i] is stored the number of groups with: 2^(2-1) < size of group < 2^(2+1)
//
//This Kernel must be launched with only one block!.
//
//Author: Simon Grimm
//March  2016
// ****************************************




// less that 512 bodies




// **********************************************************
// This kernel writes a list of close encounter pairs needed for the symplectic sub step
// Date: March 2020
// Author: Simon Grimm
// **********************************************************
void setEnc3_cpu(int N, int *Nencpairs3_h, int *Encpairs3_h, int2 *scan_h, const int NencMax){
	int id = 0 * 1 + 0;

	if(id == 0){
		Nencpairs3_h[0] = 0;
	}

	for(id = 0 * 1 + 0; id < N; ++id){
		Encpairs3_h[id * NencMax] = 0;		//Encounter pairs per body
		Encpairs3_h[id * NencMax + 1] = -1;	//list of indices	 
		Encpairs3_h[id * NencMax + 2] = 0;	//number of pairs with real gravitational influence
		Encpairs3_h[id * NencMax + 3] = 0;	//helper array for stream compaction, replace with scan_h

		scan_h[id].x = 0;
		scan_h[id].y = 0;
	}
}


// **********************************************************
// This kernel writes lists of encounter pairs for each bodies.
// It prepares the helper array for stream compaction, for a list of all involved particles
// Date: March 2020
// Author: Simon Grimm
// **********************************************************
void groupS2_cpu(int *Nencpairs2_h, int2 *Encpairs2_h, int *Nencpairs3_h, int *Encpairs3_h, int2 *scan_h, const int NencMax, const int UseTestParticles, const int N, const int SLevel){

	int id = 0 * 1 + 0;

	int Ne = Nencpairs2_h[0];
	for(id = 0 * 1 + 0; id < Ne; ++id){
		int ii = Encpairs2_h[id].x;
		int jj = Encpairs2_h[id].y;

		//count encounter pairs per body
#if def_CPU == 0
		//int NI = atomicAdd(&Encpairs3_h[ii * NencMax], 1);
		//int NJ = atomicAdd(&Encpairs3_h[jj * NencMax], 1);
		atomicAdd(&Encpairs3_h[ii * NencMax], 1);
		atomicAdd(&Encpairs3_h[jj * NencMax], 1);
#else
		//int NI = Encpairs3_h[ii * NencMax]++;
		//int NJ = Encpairs3_h[jj * NencMax]++;
		#pragma omp atomic
		Encpairs3_h[ii * NencMax]++;
		#pragma omp atomic
		Encpairs3_h[jj * NencMax]++;

#endif
//printf("group S %d %d %d %d %d\n", id, ii, jj, NI, NJ);

		//fill helper array for stream compaction
		scan_h[ii].x = 1;
		scan_h[jj].x = 1;

		if(jj < N || (UseTestParticles == 2 && ii < N)){
#if def_CPU == 0
			int Ni = atomicAdd(&Encpairs3_h[ii * NencMax + 2], 1);
#else
			int Ni;
			#pragma omp atomic capture
			Ni = Encpairs3_h[ii * NencMax + 2]++;
#endif
			Encpairs3_h[ii * NencMax + Ni + 4] = jj;
		}

		if(ii < N || (UseTestParticles == 2 && jj < N)){
#if def_CPU == 0
			int Nj = atomicAdd(&Encpairs3_h[jj * NencMax + 2], 1);
#else
			int Nj;
			#pragma omp atomic capture
			Nj = Encpairs3_h[jj * NencMax + 2]++;

#endif
			Encpairs3_h[jj * NencMax + Nj + 4] = ii;
		}
	}
}


#endif
