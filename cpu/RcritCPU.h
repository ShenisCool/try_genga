// **************************************
//This kernel calculates the critical radius rcrit = max(n1 * Rh, n2 * dt * v), with the 
//Hill radius Rh = r * (m/(3Msun))^1/3, the velocity v and two constants n1 and  n2.
//rcritv is used for the the prechecker.
//In Rh we use the radius instead of the semi major axis.
//It searches also for ejections.
//
//Authors: Simon Grimm
//November 2016
//****************************************/
void Rcritb_cpu(double4 * x4_h, double4 * v4_h, double4 *  x4b_h, double4 * v4b_h, double4 * spin_h, double4 * spinb_h, double iMsun3, double * rcrit_h, double * rcritb_h, double * rcritv_h, double * rcritvb_h, int *  index_h, int *   indexb_h, double dt, double n1, double n2, double *time_h, double time, int *EjectionFlag_m, const int N, const int NconstT, const int SLevels, const int f){
	
	int id = 0 * 1 + 0;
	
	double4 x4i;
	double4 v4i;
	
	double rcrit, rcritv;
	double rsq, vsq, r, v;
	
	if(id == 0) time_h[0] = time;
	
	for(id = 0 * 1 + 0; id < N; ++id){
		
		if(StopAtCollision_c[0] != 0 || CollTshift_c[0] != 1.0){
			//printf("Rcrit %d %g %g\n", StopAtCollision_c[0], StopMinMass_c[0], CollTshift_c[0]);
			if(f == 0){
				x4i = x4_h[id];
				v4i = v4_h[id];
				
				//store coordinates backup		
				x4b_h[id] = x4i;
				v4b_h[id] = v4i;
				spinb_h[id] = spin_h[id];
				for(int l = 0; l < SLevels; ++l){
					rcritb_h[id + l * NconstT] = rcrit_h[id + l * NconstT];
					rcritvb_h[id + l * NconstT] = rcritv_h[id + l * NconstT];
				}
				rcritb_h[id] = rcrit_h[id];
				rcritvb_h[id] = rcritv_h[id];
				indexb_h[id] = index_h[id];
			}
			else{
				//restore old coordinates
				x4i = x4b_h[id];
				v4i = v4b_h[id];
				
				x4_h[id] = x4i;
				v4_h[id] = v4i;
				spin_h[id] = spinb_h[id];
				for(int l = 0; l < SLevels; ++l){
					rcrit_h[id + l * NconstT] = rcritb_h[id + l * NconstT];
					rcritv_h[id + l * NconstT] = rcritvb_h[id + l * NconstT];
				}
				index_h[id] = indexb_h[id];
			}
		}
		else{
			x4i = x4_h[id];
			v4i = v4_h[id];
			#if def_TTV > 0
			v4b_h[id] = v4i;
			#endif
		}
		rsq = x4i.x*x4i.x + x4i.y*x4i.y + x4i.z*x4i.z + 1.0e-30;
		vsq = v4i.x*v4i.x + v4i.y*v4i.y + v4i.z*v4i.z + 1.0e-30;
		
		r = sqrt(rsq);
		v = sqrt(vsq);
		
		rcrit = n1 * r * cbrt(x4i.w * iMsun3);
		
		if(WriteEncounters_c[0] > 0){
			//in scales of planetary Radius
			double writeRadius = WriteEncountersRadius_c[0] * v4i.w;
			rcrit = fmax(rcrit, writeRadius);
		}
		
		if(StopAtEncounter_c[0] > 0){
			//rescale to non n2 rcrit 
			rcrit = StopAtEncounterRadius_c[0] * rcrit / n1;
		}

		double rc2 = n2 * fabs(dt) * v;
		rcritv = fmax(rcrit, rc2);
		
		rcrit_h[id] = rcrit;
		//the following prevents from too large critical radii for highly eccentric massive planetes
		if(rc2 > rcrit){
			rcritv_h[id] = fmax(rcritv, rcritv_h[id]);
		}
		else{
			rcritv_h[id] = rcritv;
		}
		//if(id == 3) printf("%d %g %g %g %g %g %g\n", id, rcritv_h[id], rcrit_h[id], rcritv, rc2, rcrit, r);
		//Check for Ejections or too small distances to the Sun
		if((rsq > Rcut_c[0] * Rcut_c[0] || rsq < RcutSun_c[0] * RcutSun_c[0]) && x4_h[id].w >= 0.0){
			EjectionFlag_m[0] = 1;
		}
	}
}

void Rcrit_cpu(double4 * x4_h, double4 * v4_h, double4 *  x4b_h, double4 * v4b_h, double4 * spin_h, double4 * spinb_h, double iMsun3, double * rcrit_h, double * rcritb_h, double * rcritv_h, double * rcritvb_h, int *  index_h, int *   indexb_h, double dt, double n1, double n2, double *time_h, double time, int *EjectionFlag_m, const int N, const int NconstT, const int SLevels, const int f){
	
	int id = 0 * 1 + 0;
	
	double4 x4i;
	double4 v4i;
	
	double rcrit, rcritv;
	double rsq, vsq, r, v;
	
	if(id == 0) time_h[0] = time;
	
	for(id = 0 * 1 + 0; id < N; ++id){
		
		if(StopAtCollision_c[0] != 0 || CollTshift_c[0] != 1.0){
			//printf("Rcrit %d %g %g\n", StopAtCollision_c[0], StopMinMass_c[0], CollTshift_c[0]);
			if(f == 0){
				x4i = x4_h[id];
				v4i = v4_h[id];
				
				//store coordinates backup		
				x4b_h[id] = x4i;
				v4b_h[id] = v4i;
				spinb_h[id] = spin_h[id];
				for(int l = 0; l < SLevels; ++l){
					rcritb_h[id + l * NconstT] = rcrit_h[id + l * NconstT];
					rcritvb_h[id + l * NconstT] = rcritv_h[id + l * NconstT];
				}
				indexb_h[id] = index_h[id];
			}
			else{
				//restore old coordinates
				x4i = x4b_h[id];
				v4i = v4b_h[id];
				
				x4_h[id] = x4i;
				v4_h[id] = v4i;
				spin_h[id] = spinb_h[id];
				for(int l = 0; l < SLevels; ++l){
					rcrit_h[id + l * NconstT] = rcritb_h[id + l * NconstT];
					rcritv_h[id + l * NconstT] = rcritvb_h[id + l * NconstT];
				}
				index_h[id] = indexb_h[id];
			}
		}
		else{
			x4i = x4_h[id];
			v4i = v4_h[id];
			#if def_TTV > 0
			v4b_h[id] = v4i;
			#endif
		}
		rsq = x4i.x*x4i.x + x4i.y*x4i.y + x4i.z*x4i.z + 1.0e-30;
		vsq = v4i.x*v4i.x + v4i.y*v4i.y + v4i.z*v4i.z + 1.0e-30;
		
		r = sqrt(rsq);
		v = sqrt(vsq);
		
		rcrit = n1 * r * cbrt(x4i.w * iMsun3);
		
		if(WriteEncounters_c[0] > 0){
			//in scales of planetary Radius
			double writeRadius = WriteEncountersRadius_c[0] * v4i.w;
			rcrit = fmax(rcrit, writeRadius);
		}
		
		if(StopAtEncounter_c[0] > 0){
			//rescale to non n2 rcrit 
			rcrit = StopAtEncounterRadius_c[0] * rcrit / n1;
		}
		
		double rc2 = n2 * fabs(dt) * v;
		rcritv = fmax(rcrit, rc2);
		
		rcrit_h[id] = rcrit;
		//the following prevents from too large critical radii for highly eccentric massive planets
		if(rc2 > rcrit){
			rcritv_h[id] = fmax(rcritv, rcritv_h[id]);
		}
		else{
			rcritv_h[id] = rcritv;
		}
//if(id < 10) printf("rcrit %d %g %.20g %.20g %g %g %g %g %g %d\n", id, time, x4i.x, v4i.x, x4i.w, x4b_h[id].x, x4b_h[id].w, rcritv_h[id], rcritvb_h[id], f);
		//Check for Ejections or too small distances to the Sun
		if((rsq > Rcut_c[0] * Rcut_c[0] || rsq < RcutSun_c[0] * RcutSun_c[0]) && x4_h[id].w >= 0.0){
			EjectionFlag_m[0] = 1;
		}
	}
}

//device function version of the Rcrit_kernel
/*
void Rcrit(double4 * x4_h, double4 * v4_h, double4 *  x4b_h, double4 * v4b_h, double4 * spin_h, double4 * spinb_h, double iMsun3, double * rcrit_h, double * rcritb_h, double * rcritv_h, double * rcritvb_h, int *  index_h, int *   indexb_h, double dt, double n1, double n2, int *EjectionFlag_m, const int N, const int NconstT, const int SLevels, const int f){
	
	double4 x4i;
	double4 v4i;
	
	double rcrit, rcritv;
	double rsq, vsq, r, v;
	
	
	if(id < N){
		
		if(StopAtCollision_c[0] != 0 || CollTshift_c[0] != 1.0){
			//printf("Rcrit %d %g %g\n", StopAtCollision_c[0], StopMinMass_c[0], CollTshift_c[0]);
			if(f == 0){
				x4i = x4_h[id];
				v4i = v4_h[id];
				
				//store coordinates backup		
				x4b_h[id] = x4i;
				v4b_h[id] = v4i;
				spinb_h[id] = spin_h[id];
				for(int l = 0; l < SLevels; ++l){
					rcritb_h[id + l * NconstT] = rcrit_h[id + l * NconstT];
					rcritvb_h[id + l * NconstT] = rcritv_h[id + l * NconstT];
				}
				indexb_h[id] = index_h[id];
			}
			else{
				//restore old coordinates
				x4i = x4b_h[id];
				v4i = v4b_h[id];
				
				x4_h[id] = x4i;
				v4_h[id] = v4i;
				spin_h[id] = spinb_h[id];
				for(int l = 0; l < SLevels; ++l){
					rcrit_h[id + l * NconstT] = rcritb_h[id + l * NconstT];
					rcritv_h[id + l * NconstT] = rcritvb_h[id + l * NconstT];
				}
				index_h[id] = indexb_h[id];
			}
		}
		else{
			x4i = x4_h[id];
			v4i = v4_h[id];
		}
		rsq = x4i.x*x4i.x + x4i.y*x4i.y + x4i.z*x4i.z + 1.0e-30;
		vsq = v4i.x*v4i.x + v4i.y*v4i.y + v4i.z*v4i.z + 1.0e-30;
		
		r = sqrt(rsq);
		v = sqrt(vsq);
		
		rcrit = n1 * r * cbrt(x4i.w * iMsun3);
		
		if(WriteEncounters_c[0] > 0){
			//in scales of planetary Radius
			double writeRadius = WriteEncountersRadius_c[0] * v4i.w;
			rcrit = fmax(rcrit, writeRadius);
		}
		
		if(StopAtEncounter_c[0] > 0){
			//rescale to non n2 rcrit 
			rcrit = StopAtEncounterRadius_c[0] * rcrit / n1;
		}
		
		double rc2 = n2 * fabs(dt) * v;
		rcritv = fmax(rcrit, rc2);
		
		rcrit_h[id] = rcrit;
		//the following prevents from too large critical radii for highly eccentric massive planetes
		if(rc2 > rcrit){
			rcritv_h[id] = fmax(rcritv, rcritv_h[id]);
		}
		else{
			rcritv_h[id] = rcritv;
		}
		//printf("rcrit %d %.20g %.20g %g %g %g %g %g %d\n", id, x4i.x, v4i.x, x4i.w, x4b_h[id].x, x4b_h[id].w, rcritv_h[id], rcritvb_h[id], f);
		//Check for Ejections or too small distances to the Sun
		if((rsq > Rcut_c[0] * Rcut_c[0] || rsq < RcutSun_c[0] * RcutSun_c[0]) && x4_h[id].w >= 0.0){
			EjectionFlag_m[0] = 1;
		}
	}

}
*/

// *****************************************************
// Version of the Rcrit kernel which is called from the recursive symplectic sub step method
//
// Author: Simon Grimm
// January 2019
// ********************************************************
void RcritS_cpu(double4 * x4_h, double4 * v4_h, double iMsun3, double * rcrit_h, double * rcritv_h, double dt, double n1, double n2, const int N, int *Nencpairs_h, int *Nencpairs2_h, int *Nencpairs3_h, int *Encpairs3_h, const int NencMax, const int NconstT, const int SLevel){
	
	int idd = 0 * 1 + 0;
	
	double4 x4i;
	double4 v4i;
	
	double rcrit, rcritv ;
	double rsq, vsq, r, v;
	if(idd == 0){
		Nencpairs_h[0] = 0;
		Nencpairs2_h[0] = 0;
	}
	for(idd = 0 * 1 + 0; idd < Nencpairs3_h[0]; ++idd){
		int id = Encpairs3_h[idd * NencMax + 1];
		if(id >= 0 && id < N){
			
			x4i = x4_h[id];
			v4i = v4_h[id];
			//printf("RcritS %d %d %.20g %.20g %g %g\n", idd, id, x4i.x, v4i.y, rcrit_h[id + SLevel * NconstT], rcritv_h[id + SLevel * NconstT]);			
			rsq = x4i.x*x4i.x + x4i.y*x4i.y + x4i.z*x4i.z + 1.0e-30;
			vsq = v4i.x*v4i.x + v4i.y*v4i.y + v4i.z*v4i.z + 1.0e-30;
			
			r = sqrt(rsq);
			v = sqrt(vsq);
			
			rcrit = n1 * r * cbrt(x4i.w * iMsun3);
#if def_SLn1 == 1
			rcrit = fmax(rcrit, 3.0 * v4i.w);
#endif
			double rc2 = n2 * fabs(dt) * v;
			rcritv = fmax(rcrit, rc2);
			
			if(rc2 > rcrit){
				rcritv_h[id + SLevel * NconstT] = fmax(rcritv, rcritv_h[id + SLevel * NconstT]);
			}
			else{
				rcritv_h[id + SLevel * NconstT] = rcritv;
			}
			
			rcrit_h[id + SLevel * NconstT] = rcrit;
		}
	}
}

// **************************************
//For the multi simulation mode
//This kernel calculates the critical radius rcrit = max(n1 * Rh, n2 * dt * v), with the 
//Hill radius Rh = r * (m/(3Msun))^1/3, the velocity v and two constants n1 and  n2.
//critv is used for the the prechecker.
//In Rh we use the radius instead of the semi major axis.
//It searches also for ejections.
//
//Author: Simon Grimm
//November 2016
// ****************************************


void Data::firstStep(int noColl){
#if def_CPU == 0
	if(Nst > 1){
		#if def_TTV == 2

		#else
		if(UseM3 == 0){
			firstKick_M(0, noColl);
		}
		else{
			firstKick_M3(0, noColl);
		}
		#endif
	}
	else{
		if(P.UseTestParticles > 0){
			firstKick_small(noColl);
		}
		else{
			if(NB[0] <= WarpSize){
				firstKick_16(noColl);
			}
			else{
				firstKick_largeN(noColl);
			}
		}
	}
#else
	if(P.UseTestParticles > 0){
		firstKick_small_cpu(noColl);
	}
	else{
		firstKick_cpu(noColl);
	}
#endif
}

int Data::step(int noColl){
	int er;
#if def_CPU == 0
	//Multi simulation mode
	if(MultiSim == 1){
		#if def_TTV == 2
		  er = ttv_step();
		#else

		  #if def_NoEncounters == 0
			if(UseM3 == 0){
				er = step_M(noColl);
			}
			else{
				er = step_M3(noColl);
			}
	 	  #else
			er = step_MSimple();
		  #endif
		#endif
		if(er == 0) return 0;
	}
	else{
		//Test particles
		if(P.UseTestParticles > 0){
			er = step_small(noColl);
			if(er == 0) return 0;
		}
		//check the number of massive particles
		else{
			if(NB[0] <= WarpSize){
				er =  step_16(noColl);
			}
			else{
				er = step_largeN(noColl);
			}
			if(er == 0) return 0;
		}
	}
#else
	if(P.UseTestParticles > 0){
		er = step_small_cpu(noColl);
		if(er == 0) return 0;
	}
	else{
		//step1_cpu();
		//er = 1;
		er = step_cpu(noColl);
		if(er == 0) return 0;
	}
#endif
	return 1;
}
