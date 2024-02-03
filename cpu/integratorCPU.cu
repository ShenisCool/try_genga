#include "Orbit2CPU.h"
#include "RcritCPU.h"
#include "Kick3CPU.h"
#include "HCCPU.h"
#include "FG2CPU.h"
#include "Encounter3CPU.h"
#include "BSBCPU.h"
//#include "BSBM.h"
//#include "BSBM3.h"
#include "ComEnergyCPU.h"
#include "convertCPU.h"
#include "forceCPU.h"
#include "forceYarkovskyOldCPU.h"
#include "Kick4CPU.h"
#include "BSACPU.h"
//#include "BSAM3.h"
#if def_TTV > 0
  #include "BSTTV.h"
#endif
#if def_TTV == 2
  #include "TTVAll.h"
#endif
#if def_RV == 1
  #include "BSRV.h"
#endif
#include "ScanCPU.h"
#include "createparticlesCPU.h"
#include "bvhCPU.h"
#if def_G3 > 0
	#include "BSBG3.h"
#endif


int SIn;		//Number of direction steps
int SIM;		//half of steps
double *Ct;		//time factor for HC Kick steps
double *FGt;		//time factor for Drift steps
double *Kt;		//time factor for Kick steps

int EjectionFlag2 = 0;
int StopAtEncounterFlag2 = 0;


#if def_CPU == 1
double Rcut_c[1];
double RcutSun_c[1];
int StopAtCollision_c[1];
double StopMinMass_c[1];
double CollisionPrecision_c[1]; 
double CollTshift_c[1]; 
int CollisionModel_c[1];
int2 CollTshiftpairs_c[1]; 
int WriteEncounters_c[1]; 
double WriteEncountersRadius_c[1]; 
int WriteEncountersCloudSize_c[1]; 
int StopAtEncounter_c[1]; 
double StopAtEncounterRadius_c[1]; 
double Asteroid_eps_c[1];
double Asteroid_rho_c[1];
double Asteroid_C_c[1];
double Asteroid_A_c[1];
double Asteroid_K_c[1];
double Asteroid_V_c[1];
double Asteroid_rmin_c[1];
double Asteroid_rdel_c[1];
double SolarConstant_c[1];
double Qpr_c[1];
double SolarWind_c[1];

double BSddt_c[8];		//time stepping factors in Bulirsch-Stoer method
double BSt0_c[8 * 8];

#endif


void Data::constantCopyDirectAcc(){
#if def_CPU == 0
	cudaMemcpyToSymbol(Rcut_c, &Rcut_h[0], sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(RcutSun_c, &RcutSun_h[0], sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(StopAtCollision_c, &P.StopAtCollision, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(StopMinMass_c, &P.StopMinMass, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(CollisionPrecision_c, &P.CollisionPrecision, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(CollTshift_c, &P.CollTshift, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(CollisionModel_c, &P.CollisionModel, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(WriteEncounters_c, &P.WriteEncounters, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(WriteEncountersRadius_c, &P.WriteEncountersRadius, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(WriteEncountersCloudSize_c, &P.WriteEncountersCloudSize, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(StopAtEncounter_c, &P.StopAtEncounter, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(StopAtEncounterRadius_c, &P.StopAtEncounterRadius, sizeof(double), 0, cudaMemcpyHostToDevice);

	int2 ij;
	ij.x = -1;
	ij.y = -1;
	cudaMemcpyToSymbol(CollTshiftpairs_c, &ij, sizeof(int2), 0, cudaMemcpyHostToDevice);

	cudaMemcpyToSymbol(SolarConstant_c, &P.SolarConstant, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(Qpr_c, &P.Qpr, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(SolarWind_c, &P.SolarWind, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(Asteroid_eps_c, &P.Asteroid_eps, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(Asteroid_rho_c, &P.Asteroid_rho, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(Asteroid_C_c, &P.Asteroid_C, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(Asteroid_A_c, &P.Asteroid_A, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(Asteroid_K_c, &P.Asteroid_K, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(Asteroid_V_c, &P.Asteroid_V, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(Asteroid_rmin_c, &P.Asteroid_rmin, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(Asteroid_rdel_c, &P.Asteroid_rdel, sizeof(double), 0, cudaMemcpyHostToDevice);
#else
	*Rcut_c = *Rcut_h;
	*RcutSun_c = *RcutSun_h;
	*StopAtCollision_c = P.StopAtCollision;
	*StopMinMass_c = P.StopMinMass;
	*CollisionPrecision_c = P.CollisionPrecision;
	*CollTshift_c = P.CollTshift;
	*CollisionModel_c = P.CollisionModel;
	*WriteEncounters_c = P.WriteEncounters;
	*WriteEncountersRadius_c = P.WriteEncountersRadius;
	*WriteEncountersCloudSize_c = P.WriteEncountersCloudSize;
	*StopAtEncounter_c = P.StopAtEncounter;
	*StopAtEncounterRadius_c = P.StopAtEncounterRadius;

	CollTshiftpairs_c[0].x = -1;
	CollTshiftpairs_c[0].y = -1;

	*SolarConstant_c = P.SolarConstant;
	*Qpr_c = P.Qpr;
	*SolarWind_c = P.SolarWind;
	*Asteroid_eps_c = P.Asteroid_eps;
	*Asteroid_rho_c = P.Asteroid_rho;
	*Asteroid_C_c = P.Asteroid_C;
	*Asteroid_A_c = P.Asteroid_A;
	*Asteroid_K_c = P.Asteroid_K;
	*Asteroid_V_c = P.Asteroid_V;
	*Asteroid_rmin_c = P.Asteroid_rmin;
	*Asteroid_rdel_c = P.Asteroid_rdel;
#endif
}


void Data::constantCopyBS(){

	double ddt[8];
	double t0[8 * 8];

	for(int n = 1; n <= 8; ++n){
		ddt[n-1] = 0.25 / (n*n);
	}

	for(int n = 1; n <= 8; ++n){
		for(int j = n-1; j >=1; --j){
			t0[(n-1) * 8 + (j -1)] = 1.0 / (ddt[j-1] - ddt[n-1]);
		}
	}
#if def_CPU == 0
	 cudaMemcpyToSymbol(BSddt_c, ddt, 8 * sizeof(double), 0, cudaMemcpyHostToDevice);
	 cudaMemcpyToSymbol(BSt0_c, t0, 8 * 8 * sizeof(double), 0, cudaMemcpyHostToDevice);
#else
	 memcpy(BSddt_c, ddt, 8 * sizeof(double));
	 memcpy(BSt0_c, t0, 8 * 8 * sizeof(double));
#endif
}


void initialb_cpu(int2 *Encpairs_h, int2 *Encpairs2_h, const int NBNencT){
	
	int id = 0 * 1 + 0;
	
	for(id = 0 * 1 + 0; id < NBNencT; ++id){
		Encpairs_h[id].x = -1;
		Encpairs_h[id].y = -1;
		
		Encpairs2_h[id].x = -1;
		Encpairs2_h[id].y = -1;
	}
}

/*
 void test_cpu(double4 *x4_h, double3 *a_h, int *index_h, const int N){
 
	int id = 0 * 1 + 0;
 
	for(id = 0 * 1 + 0; id < N; ++id){
		 if(fabs(a_h[id].x) > 10) printf("test %d %.g\n", id, a_h[id].x);
 	}
 }
 */


void save_cpu(double4 *x4_h, double4 *v4_h, double4 *x4bb_h, double4 *v4bb_h, double4 *spin_h, double4 *spinbb_h, double *rcrit_h, double *rcritv_h, double *rcritbb_h, double *rcritvbb_h, int *index_h, int *indexbb_h, const int N, const int NconstT, const int SLevels, const int f){

	int id = 0 * 1 + 0;

	for(id = 0 * 1 + 0; id < N; ++id){
		if(f == 1){
			x4bb_h[id] = x4_h[id];
			v4bb_h[id] = v4_h[id];
			spinbb_h[id] = spin_h[id];
			indexbb_h[id] = index_h[id];
			for(int l = 0; l < SLevels; ++l){
				rcritbb_h[id + l * NconstT] = rcrit_h[id + l * NconstT];
				rcritvbb_h[id + l * NconstT] = rcritv_h[id + l * NconstT];
			}
		}
		if(f == -1){
			x4_h[id] = x4bb_h[id];
			v4_h[id] = v4bb_h[id];
			spin_h[id] = spinbb_h[id];
			index_h[id] = indexbb_h[id];
			for(int l = 0; l < SLevels; ++l){
				rcrit_h[id + l * NconstT] = rcritbb_h[id + l * NconstT];
				rcritv_h[id + l * NconstT] = rcritvbb_h[id + l * NconstT];
			}
		}
	}
}


int Data::beforeTimeStepLoop1(){

	int er;
#if def_CPU == 0
	//(&KickEvent);	
	cudaStreamCreateWithFlags(&copyStream, cudaStreamNonBlocking);
	for(int st = 0; st < 12; ++st){
		cudaStreamCreate(&BSStream[st]);
	}
	for(int st = 0; st < 16; ++st){
		cudaStreamCreate(&hstream[st]);
	}
#endif


	//Allocate orbit data on Host and Device
	er = AllocateOrbit();
	if(er == 0) return 0;

	//allocate mapped memory//
	er = CMallocateOrbit();
	if(er == 0) return 0;

	//copy constant memory
	constantCopyDirectAcc();
	constantCopyBS();

	//Allocate aeGride
	constantCopy2();
	if(P.UseaeGrid == 1){
		er = GridaeAlloc();
		if(er == 0) return 0;
	}
	if(P.Usegas == 1){
		GasAlloc();
	}

	//Table for fastfg//
	er = FGAlloc();
	if(er == 0) return 0;

	//initialize memory//
	er = init();
	printf("\nInitialize Memory\n");

#if def_TTV > 0
  #if MCMC_NCOV > 0
	er = readMCMC_COV();
	if(er == 0) return 0;
  #endif
#endif

	//read initial conditions//
	printf("\nRead Initial Conditions\n");
	er = ic();
	if(er == 0) return 0;
	printf("Initial Conditions OK\n");

#if USE_NAF == 1
	er = naf.alloc1(NT, N_h[0], Nsmall_h[0], Nst, P.tRestart, idt_h, ict_h, P.NAFn0, P.NAFnfreqs);
	if(er == 0) return 0;

	er = naf.alloc2(NT, N_h[0], Nsmall_h[0], Nst, GSF, P.NAFformat, P.tRestart, index_h);
	if(er == 0) return 0;
#endif


#if def_CPU == 0
	//Check warp size
	{
		cudaDeviceProp devProp;
		cudaGetDeviceProperties(&devProp, P.dev[0]);
		WarpSize = devProp.warpSize;
	}
#endif

	//remove ghost particles and reorder arrays//
	int NminFlag = remove();
	//remove stopped simulations//
	if(NminFlag > 0){
		stopSimulations();
		NminFlag = 0;
		if(Nst == 0)  return 0;
	}

	error = 0;
	if(error != 0){
		fprintf(masterfile, "Start1 error = %d = %s\n",error, "-");
		printf("Start1 error = %d = %s\n",error, "-");
		return 0;
	}
	printf("Compute initial Energy\n");

	er = firstEnergy();
	if(er == 0) return 0;



	printf("Write initial Energy\n");

	//write first output
	er = firstoutput(0);
	if(er == 0) return 0;

	if(P.IrregularOutputs == 1){
		er = firstoutput(1);
	}
	if(er == 0) return 0;
	printf("Energy OK\n");

	//read aeGrid at restart time step 
	if(P.UseaeGrid == 1){
		readGridae();
	}

	//Set Gas Disc and Gas Table
	if(P.Usegas == 1){
		printf("Set Gas Table\n");
		er = setGasDisk();
		if(er == 0) return 0;
		printf("Gas Table OK\n");
	}

	// Set Order and Coefficients of the symplectic integrator //
	SymplecticP(0);


#if def_CPU == 0
	if(Nst == 1){

		//set default kernel parameters
		FTX = 128;
		RTX = 128;
		FrTX = 128;
		KP = 1;
		KTX = 1;
		KTY = 256;
		KP2 = 1;
		KTX2 = 1;
		KTY2 = 256;
		UseAcc = 1;
		UseBVH = 1;

		FILE *tuneFile;

		if(P.doTuning == 0){

			//check if tuneParameters file exists.
			tuneFile = fopen("tuningParameters.dat", "r");
			if(tuneFile == NULL){
				printf("tuningParameters.dat file not available, use default settings\n");
				GSF[0].logfile = fopen(GSF[0].logfilename, "a");
				fprintf(GSF[0].logfile, "tuningParameters.dat file not available, use default settings\n");
				fclose(GSF[0].logfile);
			}
			else{
				printf("Read tuningParameters.dat file\n");
			
				char sp[16];	
				int er = 0;

				for(int i = 0; i < 20; ++i){
					er = fscanf(tuneFile, "%s", sp);
					if(er <= 0) break;

					if(strcmp(sp, "FTX") == 0){
						fscanf(tuneFile, "%d", &FTX);
					}
					if(strcmp(sp, "RTX") == 0){
						fscanf(tuneFile, "%d", &RTX);
					}
					if(strcmp(sp, "KP") == 0){
						fscanf(tuneFile, "%d", &KP);
					}
					if(strcmp(sp, "KTX") == 0){
						fscanf(tuneFile, "%d", &KTX);
					}
					if(strcmp(sp, "KTY") == 0){
						fscanf(tuneFile, "%d", &KTY);
					}
					if(strcmp(sp, "KP2") == 0){
						fscanf(tuneFile, "%d", &KP2);
					}
					if(strcmp(sp, "KTX2") == 0){
						fscanf(tuneFile, "%d", &KTX2);
					}
					if(strcmp(sp, "KTY2") == 0){
						fscanf(tuneFile, "%d", &KTY2);
					}
					if(strcmp(sp, "FrTX") == 0){
						fscanf(tuneFile, "%d", &FrTX);
					}
					if(strcmp(sp, "UseAcc") == 0){
						fscanf(tuneFile, "%d", &UseAcc);
					}
					if(strcmp(sp, "UseBVH") == 0){
						fscanf(tuneFile, "%d", &UseBVH);
					}
				}

				fclose(tuneFile);

				printf("FTX %d\n", FTX);
				printf("RTX %d\n", RTX);
				printf("KP %d\n", KP);
				printf("KTX %d\n", KTX);
				printf("KTY %d\n", KTY);
				printf("KP2 %d\n", KP2);
				printf("KTX2 %d\n", KTX2);
				printf("KTY2 %d\n", KTY2);
				printf("FrTX %d\n", FrTX);
				printf("UseAcc %d\n", UseAcc);
				printf("UseBVH %d\n", UseBVH);

				GSF[0].logfile = fopen(GSF[0].logfilename, "a");
				fprintf(GSF[0].logfile, "Read tuningParameters.dat file\n");
				fprintf(GSF[0].logfile, "FTX %d\n", FTX);
				fprintf(GSF[0].logfile, "RTX %d\n", RTX);
				fprintf(GSF[0].logfile, "KP %d\n", KP);
				fprintf(GSF[0].logfile, "KTX %d\n", KTX);
				fprintf(GSF[0].logfile, "KTY %d\n", KTY);
				fprintf(GSF[0].logfile, "KP2 %d\n", KP2);
				fprintf(GSF[0].logfile, "KTX2 %d\n", KTX2);
				fprintf(GSF[0].logfile, "KTY2 %d\n", KTY2);
				fprintf(GSF[0].logfile, "FrTX %d\n", FrTX);
				fprintf(GSF[0].logfile, "UseAcc %d\n", UseAcc);
				fprintf(GSF[0].logfile, "UseBVH %d\n", UseBVH);
				fclose(GSF[0].logfile);
		
			}
		}
		else{
			//Tune kernel parameters
			er = tuneFG(FTX);
			if(er == 0) return 0;
			er = tuneRcrit(RTX);
			if(er == 0) return 0;
			if(P.UseTestParticles == 0){
				er = tuneKick(0, KP, KTX, KTY);
				if(er == 0) return 0;
			}
			if(P.UseTestParticles == 1){
				er = tuneKick(1, KP, KTX, KTY);
				if(er == 0) return 0;
			}
			if(P.UseTestParticles == 2){
				er = tuneKick(1, KP, KTX, KTY);
				if(er == 0) return 0;
				er = tuneKick(2, KP2, KTX2, KTY2);
				if(er == 0) return 0;
			}
			if(ForceFlag > 0){
				er = tuneForce(FrTX);
				if(er == 0) return 0;
			}
			if(P.WriteEncounters == 2){
				er = tuneBVH(UseBVH);
				if(er == 0) return 0;
			}


			tuneFile = fopen("tuningParameters.dat", "w");
			fprintf(tuneFile, "FTX %d\n", FTX);
			fprintf(tuneFile, "RTX %d\n", RTX);
			fprintf(tuneFile, "KP %d\n", KP);
			fprintf(tuneFile, "KTX %d\n", KTX);
			fprintf(tuneFile, "KTY %d\n", KTY);
			fprintf(tuneFile, "KP2 %d\n", KP2);
			fprintf(tuneFile, "KTX2 %d\n", KTX2);
			fprintf(tuneFile, "KTY2 %d\n", KTY2);
			fprintf(tuneFile, "FrTX %d\n", FrTX);
			fprintf(tuneFile, "UseAcc %d\n", UseAcc);
			fprintf(tuneFile, "UseBVH %d\n", UseBVH);
			fclose(tuneFile);
		}
		if(P.doSLTuning == 1){
			er = tuneBS();
			if(er == 0) return 0;
		}
	}
	else{
		//Nst > 1
		//set default kernel parameters
		KTM3 = 32;
		HCTM3 = 32;
		UseM3 = 0;

		FILE *tuneFile;

		if(P.doTuning == 0){

			//check if tuneParameters file exists.
			tuneFile = fopen("tuningParameters.dat", "r");
			if(tuneFile == NULL){
				printf("tuningParameters.dat file not available, use default settings\n");
				GSF[0].logfile = fopen(GSF[0].logfilename, "a");
				fprintf(GSF[0].logfile, "tuningParameters.dat file not available, use default settings\n");
				fclose(GSF[0].logfile);
			}
			else{
				printf("Read tuningParameters.dat file\n");
			
				char sp[16];	
				int er = 0;

				for(int i = 0; i < 20; ++i){
					er = fscanf(tuneFile, "%s", sp);
					if(er <= 0) break;
					if(strcmp(sp, "KTM3") == 0){
						fscanf(tuneFile, "%d", &KTM3);
					}
					if(strcmp(sp, "HCTM3") == 0){
						fscanf(tuneFile, "%d", &HCTM3);
					}
					if(strcmp(sp, "UseM3") == 0){
						fscanf(tuneFile, "%d", &UseM3);
					}
				}
				if(NBmax > NmaxM){
					UseM3 = 1;
				}

				fclose(tuneFile);
				printf("KTM3 %d\n", KTM3);
				printf("HCTM3 %d\n", HCTM3);
				printf("UseM3 %d\n", UseM3);

				GSF[0].logfile = fopen(GSF[0].logfilename, "a");
				fprintf(GSF[0].logfile, "Read tuningParameters.dat file\n");
				fprintf(GSF[0].logfile, "KTM3 %d\n", KTM3);
				fprintf(GSF[0].logfile, "HCTM3 %d\n", HCTM3);
				fprintf(GSF[0].logfile, "UseM3 %d\n", UseM3);
				fclose(GSF[0].logfile);
			}
		}
		else{
			//Tune kernel parameters
			er = tuneKickM3(KTM3);
			if(er == 0) return 0;

			er = tuneHCM3(HCTM3);
			if(er == 0) return 0;

			if(NBmax > NmaxM){
				UseM3 = 1;
			}

			tuneFile = fopen("tuningParameters.dat", "w");
			fprintf(tuneFile, "KTM3 %d\n", KTM3);
			fprintf(tuneFile, "HCTM3 %d\n", HCTM3);
			fprintf(tuneFile, "UseM3 %d\n", UseM3);
			fclose(tuneFile);
		}
	}
#else
	if(P.doSLTuning == 1){
		er = tuneBS();
		if(er == 0) return 0;
	}
	UseBVH = 2;
#endif 

	if(Nst == 1) printf("Start integration with %d simulation\n", Nst);
	else printf("Start integration with %d simulations\n", Nst);
	error = 0;
	if(error != 0){
		fprintf(masterfile, "Start2 error = %d = %s\n",error, "-");
		printf("Start2 error = %d = %s\n",error, "-");
		return 0;
	}



	fflush(masterfile);
#if USE_NAF == 1
	//compute the x and y arrays for the naf algorithm
	int NAFstep = 0;
	naf.getnafvarsCall(x4_h, v4_h, index_h, NBS_h, vcom_h, test_h, P.NAFvars, naf.x_h, naf.y_h, Msun_h, Msun_h[0].x, NT, Nst, naf.n, NAFstep, NB[0], N_h[0], Nsmall_h[0], P.UseTestParticles);
	++NAFstep;
#endif

	return 1;

}
// *****************************************************
// This function calls all necessary steps before the time step loop

// Authors: Simon Grimm
// February 2019
// ****************************************************
int Data::beforeTimeStepLoop(int ittv){

	int er;


	if(P.setElements > 0){
		er = readSetElements();
		if(er == 0){
			return 0;
		}
	}
	if(P.CreateParticles > 0){
		er = createReadFile2();
		if(er == 0){
			return 0;
		}
	}

	firstStep(0);

	error = 0;
	if(error != 0){
		fprintf(masterfile, "first kick error = %d = %s\n",error, "-");
		printf("first kick error = %d = %s\n", error, "-");
		return 0;
	}
	else{
		if(ittv == 0) printf("first kick OK\n");
	}

	if(EncFlag_m[0] > 0){
		printf("Error: more encounters than allowed: %d %d. Increase 'Maximum encounter pairs'.\n", EncFlag_m[0], P.NencMax);
		fprintf(masterfile, "Error: more encounters than allowed: %d %d. Increase 'Maximum encounter pairs'.\n", EncFlag_m[0], P.NencMax);
		return 0;
	}


	//Print first informations about close encounter pairs
	if(ittv == 0) firstInfo();
	setStartTime();

#if def_poincareFlag == 1
	sprintf(poincarefilename, "%sPoincare%s_%.*d.dat", GSF[0].path, GSF[0].X, def_NFileNameDigits, 0);
	poincarefile = fopen(poincarefilename, "w");
#endif

	irrTimeStep = 0ll;
	irrTimeStepOut = 0ll;
	if(P.IrregularOutputs == 1){
		er = readIrregularOutputs();
		if(er == 0){
			return 0;
		}
		//skip Irregular output times which are before the simulation starts
		double starttime = (P.tRestart) * idt_h[0] + ict_h[0] * 365.25;
		for(long long int i = 0ll; i < NIrrOutputs; ++i){
			if(IrrOutputs[i] >= starttime){
				break;
			}
			++irrTimeStep;
			++irrTimeStepOut;
		}
	}
	if(P.UseTransits == 1 && ittv == 0){
		er = readTransits();
		if(er == 0){
			return 0;
		}
	}
	if(P.UseRV == 1 && ittv == 0){
		er = readRV();
		if(er == 0){
			return 0;
		}
	}


	if(P.Usegas == 2){
		er = readGasFile();
		er = readGasFile2(time_h[0] / 365.25);
		if(er == 0){
			return 0;
		}
	}

	bufferCount = 0;
	bufferCountIrr = 0;
	MultiSim = 0;
	if(Nst > 1) MultiSim = 1;
	interrupt = 0;

	return 1;

}


// *****************************************************
// This function calls all necessary sub steps for computing 
// one time step.
// Authors: Simon Grimm
// March 2017
// ****************************************************
int Data::timeStepLoop(int interrupted, int ittv){
	time_h[0] = timeStep * idt_h[0] + ict_h[0] * 365.25;
	
	int er;	
	er = step(0);
	if(er == 0){
		return 0;
	}
	if(doTransits == 0 && timeStep == P.tRestart + 1){
		if(P.ei == 0 || (P.ei != 0 && timeStep % P.ei != 0)){
			firstInfoB();
		}
	}
	
	if(interrupted == 1){
		printf("GENGA is interrupted by SIGINT signal at time step %lld\n", timeStep);
		fprintf(masterfile, "GENGA is interrupted by SIGINT signal at time step %lld\n", timeStep);
		interrupt = 1;
	}

	if(ErrorFlag_m[0] > 0){
		printf("Error detected, GENGA stopped\n");
		fprintf(masterfile, "Error detected, GENGA stopped\n");
		return 0;
	}

	////(KickEvent);
	//do not synchronize here, to save time. but that means that the error message could be delayed
	//Check for too many encounters
	if(EncFlag_m[0] > 0){
		printf("Error: more encounters than allowed: %d %d. Increase 'Maximum encounter pairs'.\n", EncFlag_m[0], P.NencMax);
		fprintf(masterfile, "Error: more encounters than allowed: %d %d. Increase 'Maximum encounter pairs'.\n", EncFlag_m[0], P.NencMax);
		return 0;
	}
	
	//Check for too big groups//
	if(Nst == 1){
		er = MaxGroups();
		if(er == 0) return 0;
	}

	if(interrupt == 1){
		if(Nst == 1){
			RemoveCall();
		}
	}

	//Print Energy and log information//
	int CallEnergy = 0;
	if(interrupt == 1) CallEnergy = 1;
	if(P.ei != 0 && timeStep == P.deltaT) CallEnergy = 1;
	if(P.ci != 0 && timeStep == P.deltaT) CallEnergy = 1;
	if(P.ci > 0 && timeStep % P.ci == 0) CallEnergy = 1;

	if((P.ei > 0 && timeStep % P.ei == 0) || CallEnergy == 1){
		if(bufferCount + 1 >= P.Buffer || CallEnergy == 1){
			er = EnergyOutput(0);
			if(er == 0) return 0;
		}
	}
	
	if(P.UseaeGrid == 1){ 
		if(timeStep % 10000 == 0){
			er = copyGridae();
			if(er == 0){
				return 0;
			}
		}
	}
	//update Gas Disk
	if(P.Usegas == 2 && time_h[0] / 365.25 > GasDatatime.y){
		er = readGasFile2(time_h[0] / 365.25);
		if(er == 0){
			return 0;
		}
	}
	
//test_cpu /* 1, 16 */ (x4_h, v4_h, index_h);
	
	//Print Output//
	if((P.ci > 0 && ((timeStep - 1) % P.ci >= P.ci - P.nci)) || interrupt == 1 || (P.ci != 0 && timeStep == P.deltaT)){
		if(P.Buffer == 1){
			CoordinateOutput(0);
		}
		else if(bufferCount + 1 >= P.Buffer || interrupt == 1){
			//write out buffer
			timestepBuffer[bufferCount] = timeStep;
			for(int st = 0; st < Nst; ++st){
				NBuffer[Nst * (bufferCount) + st].x = N_h[st];
				NBuffer[Nst * (bufferCount) + st].y = Nsmall_h[st];
			}
			CoordinateToBuffer(bufferCount, 0, 0.0);
			++bufferCount;	
			CoordinateOutputBuffer(0);
		}
		else{
			//store in buffer
			timestepBuffer[bufferCount] = timeStep;
			for(int st = 0; st < Nst; ++st){
				NBuffer[Nst * (bufferCount) + st].x = N_h[st];
				NBuffer[Nst * (bufferCount) + st].y = Nsmall_h[st];
			}
			CoordinateToBuffer(bufferCount, 0, 0.0);
			++bufferCount;	
		}
		if(P.UseaeGrid == 1){
			GridaeOutput();
		}
		
#if def_poincareFlag == 1
		if((timeStep - 1) % P.ci == P.ci - P.nci){
			fclose(poincarefile);
			sprintf(poincarefilename, "%sPoincare%s_%.*lld.dat", GSF[0].path, GSF[0].X, def_NFileNameDigits, timeStep);
			//Erase old Poincare files
			poincarefile = fopen(poincarefilename, "w");
		}
#endif
	}
	
	//print irregular outputs
	if(interrupt == 1 && P.Buffer > 1){
		//write out buffer
		CoordinateOutputBuffer(1);
	}
	if(P.IrregularOutputs == 1 && irrTimeStep < NIrrOutputs && time_h[0] >= IrrOutputs[irrTimeStep]){
		
		int ni = 1; //multiple outputs per time step
		for(int i = 0; i < ni; ++i){
			double dTau = -(time_h[0] - IrrOutputs[irrTimeStep]) / idt_h[0];
			IrregularStep(dTau);
			for(int st = 0; st < Nst; ++st){
				time_h[st] += dTau * idt_h[st];
			}
			if(Nst > 1){
			}
			
			step(0);
			
			if(P.Buffer == 1){
				CoordinateOutput(1);
				int er = EnergyOutput(1);
				if(er == 0) return 0;
			}
			else if(bufferCountIrr + 1 >= P.Buffer){
				//write out buffer
				timestepBufferIrr[bufferCountIrr] = timeStep;
				for(int st = 0; st < Nst; ++st){
					NBufferIrr[Nst * (bufferCountIrr) + st].x = N_h[st];
					NBufferIrr[Nst * (bufferCountIrr) + st].y = Nsmall_h[st];
				}
				CoordinateToBuffer(bufferCountIrr, 1, dTau);
				++bufferCountIrr;
				CoordinateOutputBuffer(1);
				bufferCountIrr = 0;
				irrTimeStepOut += P.Buffer;
			}
			else{
				//store in buffer
				timestepBufferIrr[bufferCountIrr] = timeStep;
				for(int st = 0; st < Nst; ++st){
					NBufferIrr[Nst * (bufferCountIrr) + st].x = N_h[st];
					NBufferIrr[Nst * (bufferCountIrr) + st].y = Nsmall_h[st];
				}
				CoordinateToBuffer(bufferCountIrr, 1, dTau);
				++bufferCountIrr;
			}
			
			IrregularStep(-dTau);
			for(int st = 0; st < Nst; ++st){
				time_h[st] -= dTau * idt_h[st];
			}
			if(Nst > 1){
			}
			
			step(0);
			SymplecticP(1);
			
			++irrTimeStep;
			
			dTau = -(time_h[0] - IrrOutputs[irrTimeStep]) / idt_h[0];
			if(dTau <= 0) ++ni;
			
			if(ni + irrTimeStep - 1 > NIrrOutputs) break;
		}
	}
	
#if USE_NAF == 1
	//compute the x and y arrays for the naf algorithm
	if(timeStep % P.NAFinterval == 0){
		naf.getnafvarsCall(x4_h, v4_h, index_h, NBS_h, vcom_h, test_h, P.NAFvars, naf.x_h, naf.y_h, Msun_h, Msun_h[0].x, NT, Nst, naf.n, NAFstep, NB[0], N_h[0], Nsmall_h[0], P.UseTestParticles);
		++NAFstep;
		if(NAFstep % P.NAFn0 == 0){
			er = naf.nafCall(NT, N_h, N_h, Nsmall_h, Nsmall_h, Nst, GSF, time_h, time_h, idt_h, P.NAFformat, P.NAFinterval, index_h, index_h, NBS_h);
			if(er == 0) return 0;
			NAFstep = 0;
		}
	}
#endif
	// print time information //
	// this should be the last thing to print, because it is used to restart at the last possible timestep
	if((P.ci > 0 && timeStep % P.ci == 0) || interrupt == 1){
		if(bufferCount >= P.Buffer || P.Buffer == 1 || interrupt == 1){
			er = printTime(0);
			if(er == 0) return 0;
			fflush(masterfile);
			bufferCount = 0;
		}
	}
	if(interrupt == 1){
		printf("GENGA is terminated by SIGINT signal at time step %lld\n", timeStep);
		fprintf(masterfile, "GENGA is terminated by SIGINT signal at time step %lld\n", timeStep);
		return 0;
	}
	
	error = 0;
	if(error != 0){
		printf("Step error = %d = %s at time step: %lld\n",error, "-", timeStep);
		fprintf(masterfile, "Step error = %d = %s at time step: %lld\n",error, "-", timeStep);
		CoordinateOutput(4);
		return 0;
	}
	
	
	return 1;
	
}
// *****************************************************
// This function calls all necessary steps after the main loop
//
// Authors: Simon Grimm
// FEbruary 2019
// ****************************************************
int Data::Remaining(){

	int er;
#if def_CPU == 0
	//(KickEvent);	
	cudaStreamDestroy(copyStream);	
	for(int st = 0; st < 12; ++st){
		cudaStreamDestroy(BSStream[st]);
	}
	for(int st = 0; st < 16; ++st){
		cudaStreamDestroy(hstream[st]);
	}
#endif

	error = 0;
	if(error != 0){
		printf("Stream error = %d = %s %lld\n",error, "-", timeStep);
		return 0;
	}
	

	//write out the remaining buffer
	if(P.IrregularOutputs == 1){
		if(bufferCountIrr > 1){
			CoordinateOutputBuffer(1);
		}
	}
	if(bufferCount > 0){
		CoordinateOutputBuffer(0);
	}

#if def_poincareFlag == 1
	fclose(poincarefile);
#endif


	//print last informations
	printLastTime(0);
	LastInfo();

	//free all the memory on the Host and on the Device
	er = freeOrbit();
	if(er == 0) return 0;

	if(P.UseaeGrid == 1){
		free(Gridaecount_h);
	}

	if(P.Usegas == 1){
		er = freeGas();
		if(er == 0) return 0;
	}


#if USE_NAF == 1
		er = naf.naffree();
		if(er == 0) return 0;
#endif


	er = freeHost();
	if(er == 0) return 0;

	printf("GENGA terminated successfully\n");
	fprintf(masterfile, "GENGA terminated successfully\n");

	return 1;

}


// *****************************************************
// This function set the time factors fot the symplectic integrator for a given order
// The first time it must be called with E = 0, afterwards with E = 1
// Authors: Simon Grimm
// June 2015
// ****************************************************
void Data::SymplecticP(int E){
	SIn = 1;
	SIM = 1;
	double SIw[4]; //for maximal SI6
	
	//second order
	if(P.SIO == 2){
		//SI2
		SIn = 1;
		SIM = (SIn + 1) / 2;
		
		SIw[0] = 1.0;
	}
	//4th order
	//From Yoshida
	if(P.SIO == 4){
		//SI4
		SIn = 3;
		SIM = (SIn + 1) / 2;
		
		double two3r = cbrt(2.0);
		SIw[0] = - two3r / (2.0 - two3r);
		SIw[1] = 1.0 / (2.0 - two3r);
		
	}
	//6th order
	if(P.SIO == 6){
		//SI6
		SIn = 7;
		SIM = (SIn + 1) / 2;
		
		//Solution A from Yoshida
		SIw[1] = -0.117767998417887e1;
		SIw[2] = 0.235573213359357e0;
		SIw[3] = 0.784513610477560e0;
		SIw[0] = 1.0 - 2.0 * (SIw[1] + SIw[2] + SIw[3]);
	}
	if(E == 0){
		Ct = (double*)malloc(SIn*sizeof(double));
		FGt = (double*)malloc(SIn*sizeof(double));
		Kt = (double*)malloc(SIn*sizeof(double));
	}
	
	for(int sim = 0; sim < SIM; ++sim){
		FGt[sim] = SIw[SIM - sim - 1];
	}
	for(int sim = SIM; sim < SIn; ++sim){
		FGt[sim] = SIw[sim - SIM + 1];
	}
	
	for(int si = 0; si < SIn; ++si){
		Ct[si] = 0.5 * FGt[si];
	}
	for(int si = 0; si < SIn - 1; ++si){
		Kt[si] = 0.5 * (FGt[si] + FGt[si + 1]);
	}
	Kt[SIn - 1] = 0.5 * FGt[SIn - 1];

//	for(int si = 0; si <= SIn - 1; ++si){
//		printf("%d %g\n", si, Kt[si]);
//	}
}

// *****************************************************
// This function sets the time factors for an irregular output step
// dTau is the modified time step
// Authors: Simon Grimm
// June 2015
// ****************************************************
void Data::IrregularStep(double dTau){
	SIn = 1;
	SIM = 1;
	
	FGt[0] = dTau;
	Ct[0] = dTau * 0.5;
	Kt[0] = dTau * 0.5;
	
}











// *******************************************************************
// This function tests different kernel parameters for the Kick kernel, 
// and times them. It selects the fastest configuration and sets the
// values in KPP, KTX and KTY.
//
// Date: April 2019
// Author: Simon Grimm



int Data::tuneBS(){

	memset(BSstop_h, 0, sizeof(int));

	timeval start, stop;
	//(&start);
	//(&stop);
	float times;
	float timesMin = 1000000.0f;

	int L[9] =  {1, 2, 2, 2, 2,  3, 3, 3, 3};
	int LS[9] = {2, 2, 4, 8, 10, 2, 4, 8, 10};

	int LMin = 1;
	int LSMin = 2;

	GSF[0].logfile = fopen(GSF[0].logfilename, "a");
	printf("\nStarting close-encounter parameters tuning\n");
	fprintf(GSF[0].logfile, "\nStarting close-encounter parameters tuning\n");


#if def_CPU == 0
	int2 ij;
	ij.x = -1;
	ij.y = -1;
	cudaMemcpyToSymbol(CollTshiftpairs_c, &ij, sizeof(int2), 0, cudaMemcpyHostToDevice);
#else
	CollTshiftpairs_c[0].x = -1;
	CollTshiftpairs_c[0].y = -1;
#endif

	int NN = N_h[0] + Nsmall_h[0];
	//save backup values
	save_cpu /* (NN + 127) / 128, 128 */ (x4_h, v4_h, x4bb_h, v4bb_h, spin_h, spinbb_h, rcrit_h, rcritv_h, rcritbb_h, rcritvbb_h, index_h, indexbb_h, NN, NconstT, P.SLevels, 1);

	int noColl = 3;

	for(int i = 0; i < 9; ++i){

		P.SLevels = L[i];
		P.SLSteps = LS[i];

		EncpairsZeroC_cpu /* (NN + 255) / 256, 256 */ (Encpairs2_h, a_h, Nencpairs_h, Nencpairs2_h, P.NencMax, NN);
		save_cpu /* (NN + 127) / 128, 128 */ (x4_h, v4_h, x4bb_h, v4bb_h, spin_h, spinbb_h, rcrit_h, rcritv_h, rcritbb_h, rcritvbb_h, index_h, indexbb_h, NN, NconstT, P.SLevels, -1);


		Rcrit_cpu /* (NN + 255) / 256, 256 */ (x4_h, v4_h, x4b_h, v4b_h, spin_h, spinb_h, 1.0 / (3.0 * Msun_h[0].x), rcrit_h, rcritb_h, rcritv_h, rcritvb_h, index_h, indexb_h, dt_h[0], n1_h[0], n2_h[0], time_h, time_h[0], EjectionFlag_m, NN, NconstT, P.SLevels, 0);

		if(P.UseTestParticles == 0){
#if def_CPU == 0
			acc4C_cpu /* dim3( (((N_h[0] + KP - 1)/ KP) + KTX - 1) / KTX, 1, 1), dim3(KTX,KTY,1), KTX * KTY * KP * sizeof(double3) */ (x4_h, a_h, rcritv_h, Encpairs_h, Encpairs2_h, Nencpairs_h, EncFlag_m, 0, N_h[0], 0, N_h[0], P.NencMax, KP, 0);
#else
			if(Nomp == 1){
				acc4E_cpu();
			}
			else{
				acc4D_cpu();
			}
#endif
		}
		if(P.UseTestParticles == 1){
#if def_CPU == 0
			acc4C_cpu /* dim3( (((N_h[0] + Nsmall_h[0] + KP - 1)/ KP) + KTX - 1) / KTX, 1, 1), dim3(KTX,KTY,1), KTX * KTY * KP * sizeof(double3) */ (x4_h, a_h, rcritv_h, Encpairs_h, Encpairs2_h, Nencpairs_h, EncFlag_m, 0, N_h[0] + Nsmall_h[0], 0, N_h[0], P.NencMax, KP, 1);
#else
			if(Nomp == 1){
				acc4E_cpu();
				acc4Esmall_cpu();
			}
			else{
				acc4D_cpu();
				acc4Dsmall_cpu();
			}
#endif
		}
		if(P.UseTestParticles == 2){
#if def_CPU == 0
			acc4C_cpu /* dim3( (((N_h[0] + KP2 - 1)/ KP2) + KTX2 - 1) / KTX2, 1, 1), dim3(KTX2,KTY2,1), KTX2 * KTY2 * KP2 * sizeof(double3) */ (x4_h, a_h, rcritv_h, Encpairs_h, Encpairs2_h, Nencpairs_h, EncFlag_m, 0, N_h[0], N_h[0], N_h[0] + Nsmall_h[0], P.NencMax, KP2, 2);
#else
			if(Nomp == 1){
				acc4E_cpu();
				acc4Esmall_cpu();
			}
			else{
				acc4D_cpu();
				acc4Dsmall_cpu();
			}
#endif
		}


		kick32Ab_cpu /* (NN + RTX - 1) / RTX, RTX */ (x4_h, v4_h, a_h, ab_h, rcritv_h, dt_h[0] * Kt[SIn - 1] * def_ksq, Nencpairs_h, Encpairs2_h, 0, NN, P.NencMax, 1);

		HCCall(Ct[0], 1);
		fg_cpu /* (NN + FTX - 1) / FTX, FTX */ (x4_h, v4_h, xold_h, vold_h, index_h, dt_h[0] * FGt[0], Msun_h[0].x, NN, aelimits_h, aecount_h, Gridaecount_h, Gridaicount_h, 1, P.UseGR);

		printf("    Precheck-pairs:    %d\n", Nencpairs_h[0]);
		fprintf(GSF[0].logfile,"    Precheck-pairs:    %d\n", Nencpairs_h[0]);

		if(Nenc_m[0] > 0){
			for(int i = 0; i < def_GMax; ++i){
				Nenc_m[i] = 0;
			}
			Nencpairs2_h[0] = 0;		
			setNencpairs_cpu /* 1, 1 */ (Nencpairs2_h, 1);
		}

		if(EncFlag_m[0] > 0){
			printf("Error: more encounters than allowed: %d %d. Increase 'Maximum encounter pairs'.\n", EncFlag_m[0], P.NencMax);
			fprintf(masterfile, "Error: more encounters than allowed: %d %d. Increase 'Maximum encounter pairs'.\n", EncFlag_m[0], P.NencMax);
			return 0;
		}


		if(Nencpairs_h[0] > 0){
			encounter_cpu /* (Nencpairs_h[0] + 63)/ 64, 64 */ (x4_h, v4_h, xold_h, vold_h, rcrit_h, rcritv_h, dt_h[0] * FGt[0], Nencpairs_h[0], Nencpairs_h, Encpairs_h, Nencpairs2_h, Encpairs2_h, enccount_h, 1, NN, time_h[0], P.StopAtEncounter, Ncoll_m, P.MinMass);


			gettimeofday(&start, 0);

			if(P.SLevels > 1){
				if(Nencpairs2_h[0] > 0){
					double time = time_h[0];
					SEnc(time, 0, 1.0, 0, noColl);
				}
			}
			else{
				if(Nencpairs2_h[0] > 0){
#if def_CPU == 0
					groupCall();
#else
					group_cpu (Nenc_m, Nencpairs2_h, Encpairs2_h, Encpairs_h, P.NencMax, N_h[0] + Nsmall_h[0], N_h[0], P.SERIAL_GROUPING);
#endif
				}

				BSCall(0, time_h[0], noColl, 1.0);
			}

			gettimeofday(&stop, 0);

			printf("    CE:    %d; ", Nencpairs2_h[0]);
			printf("groups: %d; ", Nenc_m[0]);
			fprintf(GSF[0].logfile, "    CE:    %d; ", Nencpairs2_h[0]);
			fprintf(GSF[0].logfile, "groups: %d; ", Nenc_m[0]);
			int nn = 2;
			for(int st = 1; st < def_GMax; ++st){
				if(Nenc_m[st] > 0){
					printf("%d: %d; ", nn, Nenc_m[st]);
					fprintf(GSF[0].logfile, "%d: %d; ", nn, Nenc_m[st]);
				}
				nn *= 2;
			}
			printf("\n");
			fprintf(GSF[0].logfile, "\n");


			//(stop);
			ElapsedTime(&times, start, stop); //time in microseconds
			printf("Close-encounter time: Symplectic levels: %d, Symplectic sub steps: %d,  %.15f s\n", P.SLevels, P.SLSteps, times * 0.001);	//time in seconds
			fprintf(GSF[0].logfile, "Close-encounter time: Symplectic levels: %d, Symplectic sub steps: %d,  %.15f s\n", P.SLevels, P.SLSteps, times * 0.001);	//time in seconds
			if(times < timesMin){
				LMin = L[i];
				LSMin = LS[i];
			}
			timesMin = fmin(times, timesMin);

		}
	}
	P.SLevels = LMin;
	P.SLSteps = LSMin;

	printf("\nBest parameters: Symplectic levels: %d, Symplectic sub steps: %d,  %.15f s\n\n", P.SLevels, P.SLSteps, timesMin * 0.001);
	fprintf(GSF[0].logfile, "\nBest parameters: Symplectic levels: %d, Symplectic sub steps: %d,  %.15f s\n\n", P.SLevels, P.SLSteps, timesMin * 0.001);

	//restore old coordinate values
	EncpairsZeroC_cpu /* (NN + 255) / 256, 256 */ (Encpairs2_h, a_h, Nencpairs_h, Nencpairs2_h, P.NencMax, NN);
	save_cpu /* (NN + 127) / 128, 128 */ (x4_h, v4_h, x4bb_h, v4bb_h, spin_h, spinbb_h, rcrit_h, rcritv_h, rcritbb_h, rcritvbb_h, index_h, indexbb_h, NN, NconstT, P.SLevels, -1);

	//(start);
	//(stop);
	fclose(GSF[0].logfile);
	return 1;

}

//called during the integration, before BSCall


#if def_CPU == 1
void Data::firstKick_cpu(int noColl){
	//use last time information, the beginning of the time step
	double time = (P.tRestart + 1) * idt_h[0] + ict_h[0] * 365.25; //in the set Elements kernel, timestep wil be decreased by 1 again
	for(int i = 0; i < NconstT; ++i){
		a_h[i] = {0.0, 0.0, 0.0};
		ab_h[i] = {0.0, 0.0, 0.0};
	}
	BSstop_h[0] = 0;
	initialb_cpu (Encpairs_h, Encpairs2_h, NBNencT);

	Rcrit_cpu (x4_h, v4_h, x4b_h, v4b_h, spin_h, spinb_h, 1.0 / (3.0 * Msun_h[0].x), rcrit_h, rcritb_h, rcritv_h, rcritvb_h, index_h, indexb_h, dt_h[0], n1_h[0], n2_h[0], time_h, time, EjectionFlag_m, N_h[0], NconstT, P.SLevels, noColl);
	if(P.setElementsV == 2){ // convert barycentric velocities to heliocentric
		comCall(1);
	}
	if(P.setElementsV > 0){
		setElements_cpu (x4_h, v4_h, index_h, setElementsData_h, setElementsLine_h, Msun_h, dt_h, time_h, N_h[0], Nst, 0);
	}
	if(P.setElementsV == 2){
		comCall(-1);
	}
	if(P.KickFloat == 0){
		if(Nomp == 1){
			acc4E_cpu();
		}
		else{
			acc4D_cpu();
		}
	}
	else{
		if(Nomp == 1){
			acc4Ef_cpu();
		}
		else{
			acc4Df_cpu();
		}
	}
}
void Data::firstKick_small_cpu(int noColl){
	//use last time information, the beginning of the time step
	double time = (P.tRestart + 1) * idt_h[0] + ict_h[0] * 365.25; //in the set Elements kernel, timestep wil be decreased by 1 again
	for(int i = 0; i < NconstT; ++i){
		a_h[i] = {0.0, 0.0, 0.0};
		ab_h[i] = {0.0, 0.0, 0.0};
	}
	BSstop_h[0] = 0;
	initialb_cpu (Encpairs_h, Encpairs2_h, NBNencT);

	Rcrit_cpu (x4_h, v4_h, x4b_h, v4b_h, spin_h, spinb_h, 1.0 / (3.0 * Msun_h[0].x), rcrit_h, rcritb_h, rcritv_h, rcritvb_h, index_h, indexb_h, dt_h[0], n1_h[0], n2_h[0], time_h, time, EjectionFlag_m, N_h[0] + Nsmall_h[0], NconstT, P.SLevels, noColl);
	if(P.setElementsV == 2){ // convert barycentric velocities to heliocentric
		comCall(1);
	}
	if(P.setElementsV > 0){
		setElements_cpu (x4_h, v4_h, index_h, setElementsData_h, setElementsLine_h, Msun_h, dt_h, time_h, N_h[0], Nst, 0);
	}
	if(P.setElementsV == 2){
		comCall(-1);
	}

	if(P.KickFloat == 0){
		if(Nomp == 1){
			acc4E_cpu();
			acc4Esmall_cpu();
		}
		else{
			acc4D_cpu();
			acc4Dsmall_cpu();
		}
	}
	else{
		if(Nomp == 1){
			acc4Ef_cpu();
			acc4Efsmall_cpu();
		}
		else{
			acc4Df_cpu();
			acc4Dfsmall_cpu();
		}
	}

	if(P.SERIAL_GROUPING == 1){
		Sortb_cpu(Encpairs2_h, 0, N_h[0] + Nsmall_h[0], P.NencMax);
	}
}
#endif





void Data::BSCall(int si, double time, int noColl, double ll){
	
	time -= dt_h[0] / dayUnit;
	int N = N_h[0] + Nsmall_h[0];
	double dt = dt_h[0] / ll * FGt[si];
	
//printf(" %d | %d %d %d | %d %d %d | %d %d %d | %d %d %d\n", Nenc_m[0], Nenc_m[1], Nenc_m[2], Nenc_m[3], Nenc_m[4], Nenc_m[5], Nenc_m[6], Nenc_m[7], Nenc_m[8], Nenc_m[9],  Nenc_m[10],  Nenc_m[11],  Nenc_m[12]);
#if def_CPU == 0	
	//if(Nenc_m[1] > 0) BSB2Step_kernel <2 * 16, 2> <<< (Nenc_m[1] + 15)/ 16, 2 * 16, 0, BSStream[0] >>> (random_h, x4_h, v4_h, xold_h, vold_h, rcrit_h, rcritv_h, Encpairs2_h, dt, Msun_h[0].x, U_h, 0, index_h, Nenc_m[1], BSstop_h, Ncoll_m, Coll_h, time, spin_h, love_h, createFlag_h, aelimits_h, aecount_h, enccount_h, aecountT_h, enccountT_h, N, NconstT, NWriteEnc_m, writeEnc_h, P.UseGR, P.MinMass, P.UseTestParticles, P.SLevels, noColl);
	//if(Nenc_m[1] > 0) BSB2Step_kernel <2, 2> <<< Nenc_m[1], 2, 0, BSStream[0] >>> (random_h, x4_h, v4_h, xold_h, vold_h, rcrit_h, rcritv_h, Encpairs2_h, dt, Msun_h[0].x, U_h, 0, index_h, Nenc_m[1], BSstop_h, Ncoll_m, Coll_h, time, spin_h, love_h, createFlag_h, aelimits_h, aecount_h, enccount_h, aecountT_h, enccountT_h, N, NconstT, NWriteEnc_m, writeEnc_h, P.UseGR, P.MinMass, P.UseTestParticles, P.SLevels, noColl);

	if(Nenc_m[1] > 0) BSBStep_kernel <2, 2> <<< Nenc_m[1], 4, 0, BSStream[0] >>> (random_h, x4_h, v4_h, xold_h, vold_h, rcrit_h, rcritv_h, Encpairs2_h, dt, Msun_h[0].x, U_h, 0, index_h, BSstop_h, Ncoll_m, Coll_h, time, spin_h, love_h, createFlag_h, aelimits_h, aecount_h, enccount_h, aecountT_h, enccountT_h, N, NconstT, NWriteEnc_m, writeEnc_h, P.UseGR, P.MinMass, P.UseTestParticles, P.SLevels, noColl);
	

	if(Nenc_m[2] > 0) BSBStep_kernel <4, 4> <<< Nenc_m[2], 16, 0, BSStream[1] >>> (random_h, x4_h, v4_h, xold_h, vold_h, rcrit_h, rcritv_h, Encpairs2_h, dt, Msun_h[0].x, U_h, 1, index_h, BSstop_h, Ncoll_m, Coll_h, time, spin_h, love_h, createFlag_h, aelimits_h, aecount_h, enccount_h, aecountT_h, enccountT_h, N, NconstT, NWriteEnc_m, writeEnc_h, P.UseGR, P.MinMass, P.UseTestParticles, P.SLevels, noColl);
	if(Nenc_m[3] > 0) BSBStep_kernel <8, 8> <<< Nenc_m[3], 64, 0, BSStream[2] >>> (random_h, x4_h, v4_h, xold_h, vold_h, rcrit_h, rcritv_h, Encpairs2_h, dt, Msun_h[0].x, U_h, 2, index_h, BSstop_h, Ncoll_m, Coll_h, time, spin_h, love_h, createFlag_h, aelimits_h, aecount_h, enccount_h, aecountT_h, enccountT_h, N, NconstT, NWriteEnc_m, writeEnc_h, P.UseGR, P.MinMass, P.UseTestParticles, P.SLevels, noColl);
	if(Nenc_m[4] > 0) BSBStep_kernel <16, 16> <<< Nenc_m[4], 256, 0, BSStream[3] >>> (random_h, x4_h, v4_h, xold_h, vold_h, rcrit_h, rcritv_h, Encpairs2_h, dt, Msun_h[0].x, U_h, 3, index_h, BSstop_h, Ncoll_m, Coll_h, time, spin_h, love_h, createFlag_h, aelimits_h, aecount_h, enccount_h, aecountT_h, enccountT_h, N, NconstT, NWriteEnc_m, writeEnc_h, P.UseGR, P.MinMass, P.UseTestParticles, P.SLevels, noColl);
	if(Nenc_m[5] > 0) BSBStep_kernel <32, 8> <<< Nenc_m[5], 256, 0, BSStream[4] >>> (random_h, x4_h, v4_h, xold_h, vold_h, rcrit_h, rcritv_h, Encpairs2_h, dt, Msun_h[0].x, U_h, 4, index_h, BSstop_h, Ncoll_m, Coll_h, time, spin_h, love_h, createFlag_h, aelimits_h, aecount_h, enccount_h, aecountT_h, enccountT_h, N, NconstT, NWriteEnc_m, writeEnc_h, P.UseGR, P.MinMass, P.UseTestParticles, P.SLevels, noColl);

	/*
	if(Nenc_m[1] > 0) BSA_kernel < 2 > <<< Nenc_m[1], 2 , 0, BSStream[0] >>> (random_h, x4_h, v4_h, xold_h, vold_h, rcrit_h, rcritv_h, index_h, spin_h, love_h, createFlag_h, Encpairs_h, Encpairs2_h, dt, Msun_h[0].x, U_h, 0, N, NconstT, P.NencMax, BSstop_h, Ncoll_m, Coll_h, time, aelimits_h, aecount_h, enccount_h, aecountT_h, enccountT_h, NWriteEnc_m, writeEnc_h, P.UseGR, P.MinMass, P.UseTestParticles, P.SLevels, noColl);
	if(Nenc_m[2] > 0) BSA_kernel < 4 > <<< Nenc_m[2], 4 , 0, BSStream[1] >>> (random_h, x4_h, v4_h, xold_h, vold_h, rcrit_h, rcritv_h, index_h, spin_h, love_h, createFlag_h, Encpairs_h, Encpairs2_h, dt, Msun_h[0].x, U_h, 1, N, NconstT, P.NencMax, BSstop_h, Ncoll_m, Coll_h, time, aelimits_h, aecount_h, enccount_h, aecountT_h, enccountT_h, NWriteEnc_m, writeEnc_h, P.UseGR, P.MinMass, P.UseTestParticles, P.SLevels, noColl);
	if(Nenc_m[3] > 0) BSA_kernel < 8 > <<< Nenc_m[3], 8 , 0, BSStream[2] >>> (random_h, x4_h, v4_h, xold_h, vold_h, rcrit_h, rcritv_h, index_h, spin_h, love_h, createFlag_h, Encpairs_h, Encpairs2_h, dt, Msun_h[0].x, U_h, 2, N, NconstT, P.NencMax, BSstop_h, Ncoll_m, Coll_h, time, aelimits_h, aecount_h, enccount_h, aecountT_h, enccountT_h, NWriteEnc_m, writeEnc_h, P.UseGR, P.MinMass, P.UseTestParticles, P.SLevels, noColl);
	if(Nenc_m[4] > 0) BSA_kernel < 16 > <<< Nenc_m[4], 16 , 0, BSStream[3] >>> (random_h, x4_h, v4_h, xold_h, vold_h, rcrit_h, rcritv_h, index_h, spin_h, love_h, createFlag_h, Encpairs_h, Encpairs2_h, dt, Msun_h[0].x, U_h, 3, N, NconstT, P.NencMax, BSstop_h, Ncoll_m, Coll_h, time, aelimits_h, aecount_h, enccount_h, aecountT_h, enccountT_h, NWriteEnc_m, writeEnc_h, P.UseGR, P.MinMass, P.UseTestParticles, P.SLevels, noColl);
	if(Nenc_m[5] > 0) BSA_kernel < 32 > <<< Nenc_m[5], 32 , 0, BSStream[4] >>> (random_h, x4_h, v4_h, xold_h, vold_h, rcrit_h, rcritv_h, index_h, spin_h, love_h, createFlag_h, Encpairs_h, Encpairs2_h, dt, Msun_h[0].x, U_h, 4, N, NconstT, P.NencMax, BSstop_h, Ncoll_m, Coll_h, time, aelimits_h, aecount_h, enccount_h, aecountT_h, enccountT_h, NWriteEnc_m, writeEnc_h, P.UseGR, P.MinMass, P.UseTestParticles, P.SLevels, noColl);
	*/	

	if(Nenc_m[6] > 0) BSA_kernel < 64 > <<< Nenc_m[6], 64 , 0, BSStream[5] >>> (random_h, x4_h, v4_h, xold_h, vold_h, rcrit_h, rcritv_h, index_h, spin_h, love_h, createFlag_h, Encpairs_h, Encpairs2_h, dt, Msun_h[0].x, U_h, 5, N, NconstT, P.NencMax, BSstop_h, Ncoll_m, Coll_h, time, aelimits_h, aecount_h, enccount_h, aecountT_h, enccountT_h, NWriteEnc_m, writeEnc_h, P.UseGR, P.MinMass, P.UseTestParticles, P.SLevels, noColl);
	if(Nenc_m[7] > 0) BSA_kernel < 128 > <<< Nenc_m[7], 128 , 0, BSStream[6] >>> (random_h, x4_h, v4_h, xold_h, vold_h, rcrit_h, rcritv_h, index_h, spin_h, love_h, createFlag_h, Encpairs_h, Encpairs2_h, dt, Msun_h[0].x, U_h, 6, N, NconstT, P.NencMax, BSstop_h, Ncoll_m, Coll_h, time, aelimits_h, aecount_h, enccount_h, aecountT_h, enccountT_h, NWriteEnc_m, writeEnc_h, P.UseGR, P.MinMass, P.UseTestParticles, P.SLevels, noColl);
	if(Nenc_m[8] > 0) BSA_kernel < 256 > <<< Nenc_m[8], 256 , 0, BSStream[7] >>> (random_h, x4_h, v4_h, xold_h, vold_h, rcrit_h, rcritv_h, index_h, spin_h, love_h, createFlag_h, Encpairs_h, Encpairs2_h, dt, Msun_h[0].x, U_h, 7, N, NconstT, P.NencMax, BSstop_h, Ncoll_m, Coll_h, time, aelimits_h, aecount_h, enccount_h, aecountT_h, enccountT_h, NWriteEnc_m, writeEnc_h, P.UseGR, P.MinMass, P.UseTestParticles, P.SLevels, noColl);
 

/*		
	if(Nenc_m[1] > 0) BSA512_kernel < 2, 2 > <<< Nenc_m[1], 2 , 0, BSStream[0] >>> (random_h, x4_h, v4_h, xold_h, vold_h, xp_h, vp_h, xt_h, vt_h, rcrit_h, rcritv_h, index_h, spin_h, love_h, createFlag_h, Encpairs_h, Encpairs2_h, dx_h, dv_h, dt, Msun_h[0].x, U_h, 0, N, NconstT, P.NencMax, BSstop_h, Ncoll_m, Coll_h, time, aelimits_h, aecount_h, enccount_h, aecountT_h, enccountT_h, NWriteEnc_m, writeEnc_h, dtgr_h, P.UseGR, P.MinMass, P.UseTestParticles, P.SLevels, noColl);
	if(Nenc_m[2] > 0) BSA512_kernel < 4, 4 > <<< Nenc_m[2], 4 , 0, BSStream[0] >>> (random_h, x4_h, v4_h, xold_h, vold_h, xp_h, vp_h, xt_h, vt_h, rcrit_h, rcritv_h, index_h, spin_h, love_h, createFlag_h, Encpairs_h, Encpairs2_h, dx_h, dv_h, dt, Msun_h[0].x, U_h, 1, N, NconstT, P.NencMax, BSstop_h, Ncoll_m, Coll_h, time, aelimits_h, aecount_h, enccount_h, aecountT_h, enccountT_h, NWriteEnc_m, writeEnc_h, dtgr_h, P.UseGR, P.MinMass, P.UseTestParticles, P.SLevels, noColl);
	if(Nenc_m[3] > 0) BSA512_kernel < 8, 8 > <<< Nenc_m[3], 8 , 0, BSStream[0] >>> (random_h, x4_h, v4_h, xold_h, vold_h, xp_h, vp_h, xt_h, vt_h, rcrit_h, rcritv_h, index_h, spin_h, love_h, createFlag_h, Encpairs_h, Encpairs2_h, dx_h, dv_h, dt, Msun_h[0].x, U_h, 2, N, NconstT, P.NencMax, BSstop_h, Ncoll_m, Coll_h, time, aelimits_h, aecount_h, enccount_h, aecountT_h, enccountT_h, NWriteEnc_m, writeEnc_h, dtgr_h, P.UseGR, P.MinMass, P.UseTestParticles, P.SLevels, noColl);
	if(Nenc_m[4] > 0) BSA512_kernel < 16, 16 > <<< Nenc_m[4], 16 , 0, BSStream[0] >>> (random_h, x4_h, v4_h, xold_h, vold_h, xp_h, vp_h, xt_h, vt_h, rcrit_h, rcritv_h, index_h, spin_h, love_h, createFlag_h, Encpairs_h, Encpairs2_h, dx_h, dv_h, dt, Msun_h[0].x, U_h, 3, N, NconstT, P.NencMax, BSstop_h, Ncoll_m, Coll_h, time, aelimits_h, aecount_h, enccount_h, aecountT_h, enccountT_h, NWriteEnc_m, writeEnc_h, dtgr_h, P.UseGR, P.MinMass, P.UseTestParticles, P.SLevels, noColl);
	if(Nenc_m[5] > 0) BSA512_kernel < 32, 32 > <<< Nenc_m[5], 32 , 0, BSStream[0] >>> (random_h, x4_h, v4_h, xold_h, vold_h, xp_h, vp_h, xt_h, vt_h, rcrit_h, rcritv_h, index_h, spin_h, love_h, createFlag_h, Encpairs_h, Encpairs2_h, dx_h, dv_h, dt, Msun_h[0].x, U_h, 4, N, NconstT, P.NencMax, BSstop_h, Ncoll_m, Coll_h, time, aelimits_h, aecount_h, enccount_h, aecountT_h, enccountT_h, NWriteEnc_m, writeEnc_h, dtgr_h, P.UseGR, P.MinMass, P.UseTestParticles, P.SLevels, noColl);
	if(Nenc_m[6] > 0) BSA512_kernel < 64, 64 > <<< Nenc_m[6], 64 , 0, BSStream[0] >>> (random_h, x4_h, v4_h, xold_h, vold_h, xp_h, vp_h, xt_h, vt_h, rcrit_h, rcritv_h, index_h, spin_h, love_h, createFlag_h, Encpairs_h, Encpairs2_h, dx_h, dv_h, dt, Msun_h[0].x, U_h, 5, N, NconstT, P.NencMax, BSstop_h, Ncoll_m, Coll_h, time, aelimits_h, aecount_h, enccount_h, aecountT_h, enccountT_h, NWriteEnc_m, writeEnc_h, dtgr_h, P.UseGR, P.MinMass, P.UseTestParticles, P.SLevels, noColl);
	if(Nenc_m[7] > 0) BSA512_kernel < 128, 128 > <<< Nenc_m[7], 128 , 0, BSStream[0] >>> (random_h, x4_h, v4_h, xold_h, vold_h, xp_h, vp_h, xt_h, vt_h, rcrit_h, rcritv_h, index_h, spin_h, love_h, createFlag_h, Encpairs_h, Encpairs2_h, dx_h, dv_h, dt, Msun_h[0].x, U_h, 6, N, NconstT, P.NencMax, BSstop_h, Ncoll_m, Coll_h, time, aelimits_h, aecount_h, enccount_h, aecountT_h, enccountT_h, NWriteEnc_m, writeEnc_h, dtgr_h, P.UseGR, P.MinMass, P.UseTestParticles, P.SLevels, noColl);
	if(Nenc_m[8] > 0) BSA512_kernel < 256, 256 > <<< Nenc_m[8], 256 , 0, BSStream[0] >>> (random_h, x4_h, v4_h, xold_h, vold_h, xp_h, vp_h, xt_h, vt_h, rcrit_h, rcritv_h, index_h, spin_h, love_h, createFlag_h, Encpairs_h, Encpairs2_h, dx_h, dv_h, dt, Msun_h[0].x, U_h, 7, N, NconstT, P.NencMax, BSstop_h, Ncoll_m, Coll_h, time, aelimits_h, aecount_h, enccount_h, aecountT_h, enccountT_h, NWriteEnc_m, writeEnc_h, dtgr_h, P.UseGR, P.MinMass, P.UseTestParticles, P.SLevels, noColl);
*/	
	
	//if(Nenc_m[9] > 0) BSA512_kernel < 512, 512 > <<< Nenc_m[9], 512 , 0, BSStream[0] >>> (random_h, x4_h, v4_h, xold_h, vold_h, xp_h, vp_h, xt_h, vt_h, rcrit_h, rcritv_h, index_h, spin_h, love_h, createFlag_h, Encpairs_h, Encpairs2_h, dx_h, dv_h, dt, Msun_h[0].x, U_h, 8, N, NconstT, P.NencMax, BSstop_h, Ncoll_m, Coll_h, time, aelimits_h, aecount_h, enccount_h, aecountT_h, enccountT_h, NWriteEnc_m, writeEnc_h, dtgr_h, P.UseGR, P.MinMass, P.UseTestParticles, P.SLevels, noColl);
	

	/*
	if(Nenc_m[1] > 0) BSACall(0, 2, Nenc_m[1], si, time, FGt[si] / ll, noColl);
	if(Nenc_m[2] > 0) BSACall(1, 4, Nenc_m[2], si, time, FGt[si] / ll, noColl);
	if(Nenc_m[3] > 0) BSACall(2, 8, Nenc_m[3], si, time, FGt[si] / ll, noColl);
	if(Nenc_m[4] > 0) BSACall(3, 16, Nenc_m[4], si, time, FGt[si] / ll, noColl);
	if(Nenc_m[5] > 0) BSACall(4, 32, Nenc_m[5], si, time, FGt[si] / ll, noColl);
	if(Nenc_m[6] > 0) BSACall(5, 64, Nenc_m[6], si, time, FGt[si] / ll, noColl);
	if(Nenc_m[7] > 0) BSACall(6, 128, Nenc_m[7], si, time, FGt[si] / ll, noColl);
	if(Nenc_m[8] > 0) BSACall(7, 256, Nenc_m[8], si, time, FGt[si] / ll, noColl);
	*/	
	if(Nenc_m[9] > 0) BSACall(8, 512, Nenc_m[9], si, time, FGt[si] / ll, noColl);

	if(Nenc_m[10] > 0) BSACall(9, 1024, Nenc_m[10], si, time, FGt[si] / ll, noColl);
	if(Nenc_m[11] > 0) BSACall(10, 2048, Nenc_m[11], si, time, FGt[si] / ll, noColl);
	
	int nn = 4096;
	for(int st = 11; st < def_GMax - 1; ++st){
		if(Nenc_m[st + 1] > 0) BSACall(st, nn, Nenc_m[st + 1], si, time, FGt[si] / ll, noColl);
		nn *= 2;
	}

	//make sure that Nenc_m is updated and available on host before continue
	//for(int i = 0; i < def_GMax; ++i){
	//	Nenc_m[i] = 0;
	//}
#else

//omp task
//omp taskloop
	#pragma omp parallel
	{
	#pragma omp for nowait
	for(int idx = 0; idx < Nenc_m[1]; ++idx){ 
		BSBStep_cpu <2> (random_h, x4_h, v4_h, xold_h, vold_h, rcrit_h, rcritv_h, Encpairs2_h, dt, Msun_h[0].x, U_h, 0, index_h, BSstop_h, Ncoll_m, Coll_h, time, spin_h, love_h, createFlag_h, aelimits_h, aecount_h, enccount_h, aecountT_h, enccountT_h, N, NconstT, NWriteEnc_m, writeEnc_h, P.UseGR, P.MinMass, P.UseTestParticles, P.SLevels, noColl, idx);
	}
	#pragma omp for nowait
	for(int idx = 0; idx < Nenc_m[2]; ++idx){ 
		BSBStep_cpu <4> (random_h, x4_h, v4_h, xold_h, vold_h, rcrit_h, rcritv_h, Encpairs2_h, dt, Msun_h[0].x, U_h, 1, index_h, BSstop_h, Ncoll_m, Coll_h, time, spin_h, love_h, createFlag_h, aelimits_h, aecount_h, enccount_h, aecountT_h, enccountT_h, N, NconstT, NWriteEnc_m, writeEnc_h, P.UseGR, P.MinMass, P.UseTestParticles, P.SLevels, noColl, idx);
	}
	#pragma omp for nowait
	for(int idx = 0; idx < Nenc_m[3]; ++idx){ 
		BSBStep_cpu <8> (random_h, x4_h, v4_h, xold_h, vold_h, rcrit_h, rcritv_h, Encpairs2_h, dt, Msun_h[0].x, U_h, 2, index_h, BSstop_h, Ncoll_m, Coll_h, time, spin_h, love_h, createFlag_h, aelimits_h, aecount_h, enccount_h, aecountT_h, enccountT_h, N, NconstT, NWriteEnc_m, writeEnc_h, P.UseGR, P.MinMass, P.UseTestParticles, P.SLevels, noColl, idx);
	}
	#pragma omp for nowait
	for(int idx = 0; idx < Nenc_m[4]; ++idx){ 
		BSBStep_cpu <16> (random_h, x4_h, v4_h, xold_h, vold_h, rcrit_h, rcritv_h, Encpairs2_h, dt, Msun_h[0].x, U_h, 3, index_h, BSstop_h, Ncoll_m, Coll_h, time, spin_h, love_h, createFlag_h, aelimits_h, aecount_h, enccount_h, aecountT_h, enccountT_h, N, NconstT, NWriteEnc_m, writeEnc_h, P.UseGR, P.MinMass, P.UseTestParticles, P.SLevels, noColl, idx);
	}
	#pragma omp for nowait
	for(int idx = 0; idx < Nenc_m[5]; ++idx){ 
		BSBStep_cpu <32> (random_h, x4_h, v4_h, xold_h, vold_h, rcrit_h, rcritv_h, Encpairs2_h, dt, Msun_h[0].x, U_h, 4, index_h, BSstop_h, Ncoll_m, Coll_h, time, spin_h, love_h, createFlag_h, aelimits_h, aecount_h, enccount_h, aecountT_h, enccountT_h, N, NconstT, NWriteEnc_m, writeEnc_h, P.UseGR, P.MinMass, P.UseTestParticles, P.SLevels, noColl, idx);
	}

	/*
	#pragma omp for nowait
	for(int idx = 0; idx < Nenc_m[1]; ++idx){ 
		BSA_cpu < 2 > (random_h, x4_h, v4_h, xold_h, vold_h, rcrit_h, rcritv_h, index_h, spin_h, love_h, createFlag_h, Encpairs_h, Encpairs2_h, dt, Msun_h[0].x, U_h, 0, N, NconstT, P.NencMax, BSstop_h, Ncoll_m, Coll_h, time, aelimits_h, aecount_h, enccount_h, aecountT_h, enccountT_h, NWriteEnc_m, writeEnc_h, P.UseGR, P.MinMass, P.UseTestParticles, P.SLevels, noColl, idx);
	}
	#pragma omp for nowait
	for(int idx = 0; idx < Nenc_m[2]; ++idx){ 
		BSA_cpu < 4 > (random_h, x4_h, v4_h, xold_h, vold_h, rcrit_h, rcritv_h, index_h, spin_h, love_h, createFlag_h, Encpairs_h, Encpairs2_h, dt, Msun_h[0].x, U_h, 1, N, NconstT, P.NencMax, BSstop_h, Ncoll_m, Coll_h, time, aelimits_h, aecount_h, enccount_h, aecountT_h, enccountT_h, NWriteEnc_m, writeEnc_h, P.UseGR, P.MinMass, P.UseTestParticles, P.SLevels, noColl, idx);
	}
	#pragma omp for nowait
	for(int idx = 0; idx < Nenc_m[3]; ++idx){ 
		BSA_cpu < 8 > (random_h, x4_h, v4_h, xold_h, vold_h, rcrit_h, rcritv_h, index_h, spin_h, love_h, createFlag_h, Encpairs_h, Encpairs2_h, dt, Msun_h[0].x, U_h, 2, N, NconstT, P.NencMax, BSstop_h, Ncoll_m, Coll_h, time, aelimits_h, aecount_h, enccount_h, aecountT_h, enccountT_h, NWriteEnc_m, writeEnc_h, P.UseGR, P.MinMass, P.UseTestParticles, P.SLevels, noColl, idx);
	}
	#pragma omp for nowait
	for(int idx = 0; idx < Nenc_m[4]; ++idx){ 
		BSA_cpu < 16 > (random_h, x4_h, v4_h, xold_h, vold_h, rcrit_h, rcritv_h, index_h, spin_h, love_h, createFlag_h, Encpairs_h, Encpairs2_h, dt, Msun_h[0].x, U_h, 3, N, NconstT, P.NencMax, BSstop_h, Ncoll_m, Coll_h, time, aelimits_h, aecount_h, enccount_h, aecountT_h, enccountT_h, NWriteEnc_m, writeEnc_h, P.UseGR, P.MinMass, P.UseTestParticles, P.SLevels, noColl, idx);
	}
	#pragma omp for nowait
	for(int idx = 0; idx < Nenc_m[5]; ++idx){ 
		BSA_cpu < 32 > (random_h, x4_h, v4_h, xold_h, vold_h, rcrit_h, rcritv_h, index_h, spin_h, love_h, createFlag_h, Encpairs_h, Encpairs2_h, dt, Msun_h[0].x, U_h, 4, N, NconstT, P.NencMax, BSstop_h, Ncoll_m, Coll_h, time, aelimits_h, aecount_h, enccount_h, aecountT_h, enccountT_h, NWriteEnc_m, writeEnc_h, P.UseGR, P.MinMass, P.UseTestParticles, P.SLevels, noColl, idx);
	}
	*/


	#pragma omp for nowait
	for(int idx = 0; idx < Nenc_m[6]; ++idx){ 
		BSA_cpu < 64 > (random_h, x4_h, v4_h, xold_h, vold_h, rcrit_h, rcritv_h, index_h, spin_h, love_h, createFlag_h, Encpairs_h, Encpairs2_h, dt, Msun_h[0].x, U_h, 5, N, NconstT, P.NencMax, BSstop_h, Ncoll_m, Coll_h, time, aelimits_h, aecount_h, enccount_h, aecountT_h, enccountT_h, NWriteEnc_m, writeEnc_h, P.UseGR, P.MinMass, P.UseTestParticles, P.SLevels, noColl, idx);
	}
	#pragma omp for nowait
	for(int idx = 0; idx < Nenc_m[7]; ++idx){ 
		BSA_cpu < 128 > (random_h, x4_h, v4_h, xold_h, vold_h, rcrit_h, rcritv_h, index_h, spin_h, love_h, createFlag_h, Encpairs_h, Encpairs2_h, dt, Msun_h[0].x, U_h, 6, N, NconstT, P.NencMax, BSstop_h, Ncoll_m, Coll_h, time, aelimits_h, aecount_h, enccount_h, aecountT_h, enccountT_h, NWriteEnc_m, writeEnc_h, P.UseGR, P.MinMass, P.UseTestParticles, P.SLevels, noColl, idx);
	}
	#pragma omp for nowait
	for(int idx = 0; idx < Nenc_m[8]; ++idx){ 
		BSA_cpu < 256 > (random_h, x4_h, v4_h, xold_h, vold_h, rcrit_h, rcritv_h, index_h, spin_h, love_h, createFlag_h, Encpairs_h, Encpairs2_h, dt, Msun_h[0].x, U_h, 7, N, NconstT, P.NencMax, BSstop_h, Ncoll_m, Coll_h, time, aelimits_h, aecount_h, enccount_h, aecountT_h, enccountT_h, NWriteEnc_m, writeEnc_h, P.UseGR, P.MinMass, P.UseTestParticles, P.SLevels, noColl, idx);
	}

	}

	/*
	if(Nenc_m[1] > 0) BSACall(0, 2, Nenc_m[1], si, time, FGt[si] / ll, noColl);
	if(Nenc_m[2] > 0) BSACall(1, 4, Nenc_m[2], si, time, FGt[si] / ll, noColl);
	if(Nenc_m[3] > 0) BSACall(2, 8, Nenc_m[3], si, time, FGt[si] / ll, noColl);
	if(Nenc_m[4] > 0) BSACall(3, 16, Nenc_m[4], si, time, FGt[si] / ll, noColl);
	if(Nenc_m[5] > 0) BSACall(4, 32, Nenc_m[5], si, time, FGt[si] / ll, noColl);
	if(Nenc_m[6] > 0) BSACall(5, 64, Nenc_m[6], si, time, FGt[si] / ll, noColl);
	if(Nenc_m[7] > 0) BSACall(6, 128, Nenc_m[7], si, time, FGt[si] / ll, noColl);
	if(Nenc_m[8] > 0) BSACall(7, 256, Nenc_m[8], si, time, FGt[si] / ll, noColl);
	*/
	if(Nenc_m[9] > 0) BSACall(8, 512, Nenc_m[9], si, time, FGt[si] / ll, noColl);

	if(Nenc_m[10] > 0) BSACall(9, 1024, Nenc_m[10], si, time, FGt[si] / ll, noColl);
	if(Nenc_m[11] > 0) BSACall(10, 2048, Nenc_m[11], si, time, FGt[si] / ll, noColl);
	
	int nn = 4096;
	for(int st = 11; st < def_GMax - 1; ++st){
		if(Nenc_m[st + 1] > 0) BSACall(st, nn, Nenc_m[st + 1], si, time, FGt[si] / ll, noColl);
		nn *= 2;
	}

#endif
}



int Data::RemoveCall(){
#if def_TTV == 0
	int NminFlag = remove();
	if(NminFlag == 1){
		fprintf(masterfile, "Number of bodies smaller than Nmin, simulation stopped\n");
		printf("Number of bodies smaller than Nmin, simulation stopped\n");
		return 0;
	}
	if(NminFlag == 2){
		fprintf(masterfile, "Number of test particles smaller than NminTP, simulation stopped\n");
		printf("Number of test particles smaller than NminTP, simulation stopped\n");
		return 0;
	}
	CollisionFlag = 0;
#endif
	return 1;
}

int Data::CollisionCall(int noColl){
#if def_TTV == 0
	if(Ncoll_m[0] > def_MaxColl){
		fprintf(masterfile, "Error: More Collisions than def_MaxColl, simulation stopped\n");
		printf("Error: More Collisions than def_MaxColl, simulation stopped\n");
		return 0;
	}
	int stopAtCollision = 0;
	if(noColl == 0){
		stopAtCollision = printCollisions();
	}
	CollisionFlag = 1;

	if(noColl == 0 && stopAtCollision == 1 && (P.StopAtCollision == 1 || P.CollTshift > 1.0)){
		
printf("save  1\n");
		save_cpu /* (N_h[0] + Nsmall_h[0] + 127) / 128, 128 */ (x4_h, v4_h, x4bb_h, v4bb_h, spin_h, spinbb_h, rcrit_h, rcritv_h, rcritbb_h, rcritvbb_h, index_h, indexbb_h, N_h[0] + Nsmall_h[0], NconstT, P.SLevels, 1);
		
		if(P.CollTshift > 1.0){
			//restore old step and increase radius 
			int NOld = Ncoll_m[0];
			for(int i = 0; i < NOld; ++i){

printf("Backup step 1 %d %.20g %.20g %.20g\n", i, Coll_h[i * def_NColl] * 365.25, time_h[0] - idt_h[0], (Coll_h[i * def_NColl] * 365.25 - (time_h[0] - idt_h[0])) / idt_h[0]);
				int2 ij;
				ij.x = int(Coll_h[i * def_NColl + 1]);
				ij.y = int(Coll_h[i * def_NColl + 13]);
printf("Tshiftpairs %d %d\n", ij.x, ij.y);
#if def_CPU == 0
				cudaMemcpyToSymbol(CollTshiftpairs_c, &ij, sizeof(int2), 0, cudaMemcpyHostToDevice);
#else
				CollTshiftpairs_c[0].x = ij.x;
				CollTshiftpairs_c[0].y = ij.y;
#endif

				IrregularStep(1.0);

				int N0 = Ncoll_m[0];
				BSAstop_h[0] = 0;
				memset(BSstop_h, 0, sizeof(int));
				bStep(1);
				error = 0;
				if(error != 0){
					printf("Backup step 1  error = %d = %s\n",error, "-");
					return 0;
				}
				//If all collisions are found, the number of collisions must be doubled now

printf("N0 %d %d %d\n", NOld, N0, Ncoll_m[0]);
				if(N0 == Ncoll_m[0]){
printf("Revert time step\n");
					IrregularStep(-1.0);
printf("Backup step -1 %.20g %.20g %.20g\n", (time_h[0] - idt_h[0]) - idt_h[0], time_h[0] - idt_h[0], -1.0);
					N0 = Ncoll_m[0];
					BSAstop_h[0] = 0;
					memset(BSstop_h, 0, sizeof(int));
					bStep(-1);
					error = 0;
					if(error != 0){
						printf("Backup step -1  error = %d = %s\n",error, "-");
						return 0;
					}
					if(N0 == Ncoll_m[0]){
						printf("Error: Collision time could not be reconstructed. Maybe CollTshift is too large.\n");
						return 0;
					}

				}
				

			}
			printCollisionsTshift();
printf("print Collision Tshift\n");
		}


		if(P.StopAtCollision == 1){
			double Coltime = 1.0e100;
			for(int i = 0; i < Ncoll_m[0]; ++i){
				if(Coll_h[i * def_NColl + 2] >= P.StopMinMass && Coll_h[i * def_NColl + 14] >= P.StopMinMass){
					Coltime = fmin(Coltime, Coll_h[i * def_NColl]);
				}
 
			}
printf("Backup step 2 %.20g %.20g %.20g\n", Coltime * 365.25, time_h[0] - idt_h[0], (Coltime * 365.25 - (time_h[0] - idt_h[0])) / idt_h[0]);

			IrregularStep(1.0 * ((Coltime * 365.25 - time_h[0] + idt_h[0]) / idt_h[0]));
			BSAstop_h[0] = 0;
			memset(BSstop_h, 0, sizeof(int));
			bStep(2);
			error = 0;
			if(error != 0){
				printf("Backup step 2  error = %d = %s\n",error, "-");
				return 0;
			}

			time_h[0] = Coltime * 365.25;
			CoordinateOutput(2);
			P.ci = -1;
			return 0;
		}

		Ncoll_m[0] = 0;
		BSAstop_h[0] = 0;
		memset(BSstop_h, 0, sizeof(int));
	
		// P.StopAtCollision = 0
printf("save -1\n");
		save_cpu /* (N_h[0] + Nsmall_h[0] + 127) / 128, 128 */ (x4_h, v4_h, x4bb_h, v4bb_h, spin_h, spinbb_h, rcrit_h, rcritv_h, rcritbb_h, rcritvbb_h, index_h, indexbb_h, N_h[0] + Nsmall_h[0], NconstT, P.SLevels, -1);
		IrregularStep(1.0);
		return 1;
	}
	else{
		Ncoll_m[0] = 0;
		return 1;
	}
#else

	return 1;
#endif
}


int Data::writeEncCall(){
	int er = printEncounters();
	if(er == 0){
		return 0;
	}
	NWriteEnc_m[0] = 0;
	return 1;
}

int Data::EjectionCall(){
#if def_TTV == 0
	Ejection();
	EjectionFlag_m[0] = 0;
	EjectionFlag2 = 1;
	int NminFlag = remove();
	if(NminFlag > 0){
		//at least one simulation has less bodies than Nmin and must be stopped
		
		if(P.ci != 0){
			CoordinateOutput(3);
		}
		
		stopSimulations();
		if(Nst == 0){
			return 0;
		}
	}
#endif
	return 1;
}

int Data::StopAtEncounterCall(){
#if def_TTV == 0
	
	if(Nst == 1){
		n1_h[0] = -1;
		
	}
	else{
	}
	if(P.ci != 0){
		CoordinateOutput(3);
	}
	stopSimulations();
	if(Nst == 0){
		return 0;
	}
#endif
	return 1;
}



// ******************************************
// This fucntions calls the PoincareSection kernel
// It prints the section of surface: time, particle ID, x, v, to the file Poincare_X.dat
//Authors: Simon Grimm, Joachim Stadel
//March 2014
// *******************************************
#if def_poincareFlag == 1
int Data::PoincareSectionCall(double t){
	if(SIn > 1){
		printf("Compute Poincare Sections only with the second Order integrator!\n");
		fprintf(masterfile, "Compute Poincare Sections only with the second Order integrator!\n");
		return 0;
	}
	PoincareSection_cpu /* (N_h[0] + 255) / 256, 256 */ (x4_h, v4_h, xold_h, vold_h, index_h, Msun_h[0].x, N_h[0], 0, PFlag_h);
	
	if(PFlag_h[0] == 1){
		memcpy(x4_h, xold_h, N_h[0] * sizeof(double4));
		memcpy(v4_h, vold_h, N_h[0] * sizeof(double4));
		for(int i = 0; i < N_h[0]; ++i){
			if(v4_h[i].w < 0.0 && x4_h[i].w >= 0.0){
				fprintf(poincarefile, "%.16g %d %g %g\n", t/365.25, index_h[i], x4_h[i].x, v4_h[i].x);
				
			}
		}
		PFlag_h[0] = 0; 
	}
	return 1;
}
#endif

//Recursive symplectic close encounter Step.
//At the last level BS is called
//noColl == 3 is used in tuning step
//noColl == 1 or -1 is used in collision precision backtracing
void Data::SEnc(double &time, int SLevel, double ll, int si, int noColl){

#if def_CPU == 0
	int nt = min(N_h[0] + Nsmall_h[0], 512);
	int nb = (N_h[0] + Nsmall_h[0] + nt - 1) / nt;
#endif
	if((noColl == 1 || noColl == -1) && BSAstop_h[0] == 3){
		printf("stop SEnc call b\n");
		return; 
	}

//printf("SEnc %d %d %d\n", SLevel, Nencpairs_h[0], Nencpairs2_h[0]);	
	if(noColl == 0 || SLevel > 0 || noColl == 3){	
		int NN = N_h[0] + Nsmall_h[0];
		setEnc3_cpu /* nb, nt */ (NN, Nencpairs3_h + SLevel, Encpairs3_h + SLevel * NBNencT, scan_h, P.NencMax);
		groupS2_cpu /* (Nencpairs2_h[0] + 511) / 512, 512 */ (Nencpairs2_h, Encpairs2_h, Nencpairs3_h + SLevel, Encpairs3_h + SLevel * NBNencT, scan_h, P.NencMax, P.UseTestParticles, N_h[0], SLevel);	

#if def_CPU == 0
		if(NN <= WarpSize){
			Scan32c_kernel <<< 1, WarpSize >>> (scan_h, Encpairs3_h + SLevel * NBNencT, Nencpairs3_h + SLevel, NN, P.NencMax);

		}
		else if(NN <= 1024){
			int nn = (NN + WarpSize - 1) / WarpSize;
			Scan32a_kernel <<< 1, nn * WarpSize, WarpSize * sizeof(int) >>> (scan_h, Encpairs3_h + SLevel * NBNencT, Nencpairs3_h + SLevel, NN, P.NencMax);
		}
		else{
			int nct = 1024;
			int ncb = min((NN + nct - 1) / nct, 1024);

			Scan32d1_kernel <<< ncb, nct, WarpSize * sizeof(int) >>> (scan_h, NN);
			Scan32d2_kernel <<< 1, ((ncb + WarpSize - 1) / WarpSize) * WarpSize, WarpSize * sizeof(int)  >>> (scan_h, NN);
			Scan32d3_kernel  <<< ncb, nct >>>  (Encpairs3_h + SLevel * NBNencT, scan_h, Nencpairs3_h + SLevel, NN, P.NencMax);
		}
#else
		Scan_cpu(scan_h, Encpairs3_h + SLevel * NBNencT, Nencpairs3_h + SLevel, NN, P.NencMax);
#endif

	}
// /*if(timeStep % 1000 == 0) */printf("Base0 %d %d %d\n", Nencpairs_h[0], Nencpairs2_h[0], Nencpairs3_h[SLevel]);

#if def_CPU == 0
	int nt3 = min(Nencpairs3_h[SLevel], 512);
	int nb3 = (Nencpairs3_h[SLevel] + nt3 - 1) / nt3;
	int ntf3 = min(Nencpairs3_h[SLevel], 128);
	int nbf3 = (Nencpairs3_h[SLevel] + ntf3 - 1) / ntf3;
#endif
	if(P.SERIAL_GROUPING == 1){
		if(noColl == 0 || SLevel > 0 || noColl == 3){	
			SortSb_cpu /* nb3, nt3 */ (Encpairs3_h + SLevel * NBNencT, Nencpairs3_h + SLevel, N_h[0] + Nsmall_h[0], P.NencMax);
		}
	}
	
	SLevel += 1;
	double l = P.SLSteps;	//number of sub steps
	ll *= l;
	//loop over sub time steps
	for(int s = 0; s < l; ++s){
// /*if(timeStep % 1000 == 0)*/ printf("Level %d %d %g %g\n", SLevel, s, ll, time);
		
		if(s == 0){
#if def_SLn1 == 0
			RcritS_cpu /* nb3, nt3 */  (xold_h, vold_h, 1.0 / (3.0 * Msun_h[0].x), rcrit_h, rcritv_h, dt_h[0] / ll, n1_h[0], n2_h[0], N_h[0] + Nsmall_h[0], Nencpairs_h, Nencpairs2_h, Nencpairs3_h + SLevel - 1, Encpairs3_h + (SLevel - 1) * NBNencT, P.NencMax, NconstT, SLevel);
#else
			RcritS_cpu /* nb3, nt3 */  (xold_h, vold_h, 1.0 / (3.0 * Msun_h[0].x), rcrit_h, rcritv_h, dt_h[0] / ll, n1_h[0] / ll, n2_h[0], N_h[0] + Nsmall_h[0], Nencpairs_h, Nencpairs2_h, Nencpairs3_h + SLevel - 1, Encpairs3_h + (SLevel - 1) * NBNencT, P.NencMax, NconstT, SLevel);
#endif
			kickS_cpu /* nb3, nt3 */ (x4_h, v4_h, xold_h, vold_h, rcritv_h, dt_h[0] / ll * Kt[SIn - 1] * def_ksq, Nencpairs_h, Encpairs_h, Encpairs2_h, Nencpairs3_h + SLevel - 1, Encpairs3_h + (SLevel - 1) * NBNencT, N_h[0] + Nsmall_h[0], NconstT, P.NencMax, SLevel, P.SLevels, 0);
		}
		else{
#if def_SLn1 == 0
			RcritS_cpu /* nb3, nt3 */  (x4_h, v4_h, 1.0 / (3.0 * Msun_h[0].x), rcrit_h, rcritv_h, dt_h[0] / ll, n1_h[0], n2_h[0], N_h[0] + Nsmall_h[0], Nencpairs_h, Nencpairs2_h, Nencpairs3_h + SLevel - 1, Encpairs3_h + (SLevel - 1) * NBNencT, P.NencMax, NconstT, SLevel);
#else
			RcritS_cpu /* nb3, nt3 */  (x4_h, v4_h, 1.0 / (3.0 * Msun_h[0].x), rcrit_h, rcritv_h, dt_h[0] / ll, n1_h[0] / ll, n2_h[0], N_h[0] + Nsmall_h[0], Nencpairs_h, Nencpairs2_h, Nencpairs3_h + SLevel - 1, Encpairs3_h + (SLevel - 1) * NBNencT, P.NencMax, NconstT, SLevel);
#endif
			kickS_cpu /* nb3, nt3 */ (x4_h, v4_h, x4_h, v4_h, rcritv_h, dt_h[0] / ll * Kt[SIn - 1] * def_ksq, Nencpairs_h, Encpairs_h, Encpairs2_h, Nencpairs3_h + SLevel - 1, Encpairs3_h + (SLevel - 1) * NBNencT, N_h[0] + Nsmall_h[0], NconstT, P.NencMax, SLevel, P.SLevels, 2);
		}
// /*if(timeStep % 1000 == 0) */printf("Nencpairs %d\n", Nencpairs_h[0]);
		fgS_cpu /* nbf3, ntf3 */ (x4_h, v4_h, xold_h, vold_h, index_h, dt_h[0] / ll * FGt[si], Msun_h[0].x, N_h[0] + Nsmall_h[0], aelimits_h, aecount_h, Gridaecount_h, Gridaicount_h, 1, P.UseGR, Nencpairs3_h + SLevel - 1, Encpairs3_h + (SLevel - 1) * NBNencT, P.NencMax);
		if(Nencpairs_h[0] > 0){
			encounter_cpu /* (Nencpairs_h[0] + 63)/ 64, 64 */ (x4_h, v4_h, xold_h, vold_h, rcrit_h + SLevel * NconstT, rcritv_h + SLevel * NconstT, dt_h[0] / ll * FGt[si], Nencpairs_h[0], Nencpairs_h, Encpairs_h, Nencpairs2_h, Encpairs2_h, enccount_h, 1, N_h[0] + Nsmall_h[0], time, P.StopAtEncounter, Ncoll_m, P.MinMass);
// /*if(timeStep % 1000 == 0) */printf("Nencpairs2: %d %d n1: %g\n", Nencpairs_h[0], Nencpairs2_h[0], n1_h[0] / ll);
			if(Nencpairs2_h[0] > 0){
				
				if(SLevel < P.SLevels - 1){
					SEnc(time, SLevel, ll, si, noColl);
				}
				else{
#if def_CPU == 0
					groupCall();
#else
					group_cpu (Nenc_m, Nencpairs2_h, Encpairs2_h, Encpairs_h, P.NencMax, N_h[0] + Nsmall_h[0], N_h[0], P.SERIAL_GROUPING);
#endif


					BSCall(si, time, noColl, ll);
					time += dt_h[0] / ll / dayUnit;

					if(noColl == 1 || noColl == -1){
						if(BSAstop_h[0] == 3){
							printf("stop SEnc call\n");
							return; 
						}
					}
				}
			}
		}
		kickS_cpu /* nb3, nt3 */ (x4_h, v4_h, x4_h, v4_h, rcritv_h, dt_h[0] / ll * Kt[SIn - 1] * def_ksq, Nencpairs_h, Encpairs_h, Encpairs2_h, Nencpairs3_h + SLevel - 1, Encpairs3_h + (SLevel - 1) * NBNencT, N_h[0] + Nsmall_h[0], NconstT, P.NencMax, SLevel, P.SLevels, 1);
	}
} 

#if def_CPU == 1
int Data::bStep(int noColl){
	Rcrit_cpu (x4_h, v4_h, x4b_h, v4b_h, spin_h, spinb_h, 1.0 / (3.0 * Msun_h[0].x), rcrit_h, rcritb_h, rcritv_h, rcritvb_h, index_h, indexb_h, dt_h[0], n1_h[0], n2_h[0], time_h, time_h[0], EjectionFlag_m, N_h[0] + Nsmall_h[0], NconstT, P.SLevels, noColl);
	kick32C_cpu (x4_h, v4_h, ab_h, N_h[0] + Nsmall_h[0], dt_h[0] * Kt[0]);
	HCCall(Ct[0], 1);
	fg_cpu (x4_h, v4_h, xold_h, vold_h, index_h, dt_h[0] * FGt[0], Msun_h[0].x, N_h[0] + Nsmall_h[0], aelimits_h, aecount_h, Gridaecount_h, Gridaicount_h, 0, P.UseGR);

	if(P.SLevels > 1){
		if(Nencpairs2_h[0] > 0){
			double time = time_h[0];
			SEnc(time, 0, 1.0, 0, noColl);
		}
	}
	else{
		BSCall(0, time_h[0], noColl, 1.0);
	}


	HCCall(Ct[0], -1);
	
	kick32C_cpu (x4_h, v4_h, ab_h, N_h[0] + Nsmall_h[0], dt_h[0] * Kt[0]);

	return 0;
}
#else
int Data::bStep(int noColl){
	Rcrit_cpu /* (N_h[0] + Nsmall_h[0] + RTX - 1) / RTX, RTX */ (x4_h, v4_h, x4b_h, v4b_h, spin_h, spinb_h, 1.0 / (3.0 * Msun_h[0].x), rcrit_h, rcritb_h, rcritv_h, rcritvb_h, index_h, indexb_h, dt_h[0], n1_h[0], n2_h[0], time_h, time_h[0], EjectionFlag_m, N_h[0] + Nsmall_h[0], NconstT, P.SLevels, noColl);
	kick32C_kernel <<< (N_h[0] + Nsmall_h[0] + 127) / 128, 128 >>> (x4_h, v4_h, ab_h, N_h[0] + Nsmall_h[0], dt_h[0] * Kt[0]);
	HCCall(Ct[0], 1);
	fg_cpu /*(N_h[0] + Nsmall_h[0] + FTX - 1)/FTX, FTX */ (x4_h, v4_h, xold_h, vold_h, index_h, dt_h[0] * FGt[0], Msun_h[0].x, N_h[0] + Nsmall_h[0], aelimits_h, aecount_h, Gridaecount_h, Gridaicount_h, 0, P.UseGR);

	if(P.SLevels > 1){
		if(Nencpairs2_h[0] > 0){
			double time = time_h[0];
			SEnc(time, 0, 1.0, 0, noColl);
		}
	}
	else{
		BSCall(0, time_h[0], noColl, 1.0);
	}


	HCCall(Ct[0], -1);
	
	kick32C_kernel <<< (N_h[0] + Nsmall_h[0] + 127) / 128, 128 >>> (x4_h, v4_h, ab_h, N_h[0] + Nsmall_h[0], dt_h[0] * Kt[0]);

	return 0;
}
#endif



#if def_CPU == 1
// *************************************************
// Step CPU
// *************************************************
void Data::step1_cpu(){

	Rcrit_cpu (x4_h, v4_h, x4b_h, v4b_h, spin_h, spinb_h, 1.0 / (3.0 * Msun_h[0].x), rcrit_h, rcritb_h, rcritv_h, rcritvb_h, index_h, indexb_h, dt_h[0], n1_h[0], n2_h[0], time_h, time_h[0], EjectionFlag_m, N_h[0], NconstT, P.SLevels, 0);

	double Msun = Msun_h[0].x;
	double dt05 = dt_h[0] * 0.5;
	double dt05Msun = dt05 / Msun;

	//Kick  
	kick32Ab_cpu (x4_h, v4_h, a_h, ab_h, rcritv_h, dt_h[0] * Kt[SIn - 1] * def_ksq, Nencpairs_h, Encpairs2_h, 0, N_h[0], P.NencMax, 1);
	

	for(int si = 0; si < SIn; ++si){
		
		//HC
/*		double3 a = {0.0, 0.0, 0.0};

		//#pragma omp parallel for
		for(int i = 0; i < N_h[0]; ++i){
			a.x += x4_h[i].w * v4_h[i].x;
			a.y += x4_h[i].w * v4_h[i].y;
			a.z += x4_h[i].w * v4_h[i].z;
		//printf("HCA %d %d %g %g %g %g\n", id, j, v4.x, v4j.x, mj, a.x);
		}

		a.x *= dt05Msun;
		a.y *= dt05Msun;
		a.z *= dt05Msun;


		for(int i = 0; i < N_h[0]; ++i){
			x4_h[i].x += a.x;
			x4_h[i].y += a.y;
			x4_h[i].z += a.z;
		}
*/
		HCCall(Ct[si], 1);

		//FG

		#pragma omp parallel for 
		for(int i = 0; i < N_h[0]; ++i){
			unsigned int aecount = 0u;
			xold_h[i] = x4_h[i];
			vold_h[i] = v4_h[i];
			int index = index_h[i];
			float4 aelimits = aelimits_h[i];
			//fgcfull(x4_h[i], v4_h[i], dt, mu, P.UseGR);
			fgfull(x4_h[i], v4_h[i], dt_h[0], def_ksq * Msun, Msun, aelimits, aecount, Gridaecount_h, Gridaicount_h, 0, i, index, P.UseGR);
			aecount_h[i] += aecount;
		}

//                fg_cpu (x4_h, v4_h, xold_h, vold_h, index_h, dt_h[0] * FGt[si], Msun_h[0].x, N_h[0], aelimits_h, aecount_h, Gridaecount_h, Gridaicount_h, si, P.UseGR);

		//HC
	/*	a = {0.0, 0.0, 0.0};
		for(int j = 0; j < N_h[0]; ++j){
			a.x += x4_h[j].w * v4_h[j].x;
			a.y += x4_h[j].w * v4_h[j].y;
			a.z += x4_h[j].w * v4_h[j].z;
		//printf("HCA %d %d %g %g %g %g\n", id, j, v4.x, v4j.x, mj, a.x);
		}

		a.x *= dt05Msun;
		a.y *= dt05Msun;
		a.z *= dt05Msun;

		for(int i = 0; i < N_h[0]; ++i){
			x4_h[i].x += a.x;
			x4_h[i].y += a.y;
			x4_h[i].z += a.z;
		}
	*/
		HCCall(Ct[si], -1);
	}

	if(Nomp == 1){
		acc4E_cpu();
	}
	else{
		acc4D_cpu();
	}
	kick32Ab_cpu (x4_h, v4_h, a_h, ab_h, rcritv_h, dt_h[0] * Kt[SIn - 1] * def_ksq, Nencpairs_h, Encpairs2_h, 0, N_h[0], P.NencMax, 1);

}


int Data::step_cpu(int noColl){
	Rcrit_cpu (x4_h, v4_h, x4b_h, v4b_h, spin_h, spinb_h, 1.0 / (3.0 * Msun_h[0].x), rcrit_h, rcritb_h, rcritv_h, rcritvb_h, index_h, indexb_h, dt_h[0], n1_h[0], n2_h[0], time_h, time_h[0], EjectionFlag_m, N_h[0], NconstT, P.SLevels, noColl);

	//use last time step information for setElements function, the beginning of the time step
	if(P.setElementsV == 2){ // convert barycentric velocities to heliocentric
		comCall(1);
	}
	if(P.setElementsV > 0){
		setElements_cpu(x4_h, v4_h, index_h, setElementsData_h, setElementsLine_h, Msun_h, dt_h, time_h, N_h[0], Nst, 0);
	}
	if(P.setElementsV == 2){
		comCall(-1);
	}

	if(P.SERIAL_GROUPING == 1){
		Sortb_cpu(Encpairs2_h, 0, N_h[0], P.NencMax);
	}
	if(doTransits == 0){
		if(EjectionFlag2 == 0){
			kick32Ab_cpu(x4_h, v4_h, a_h, ab_h, rcritv_h, dt_h[0] * Kt[SIn - 1] * def_ksq, Nencpairs_h, Encpairs2_h, 0, N_h[0], P.NencMax, 1);
		}
		else{
			if(P.KickFloat == 0){
				if(Nomp == 1){
					acc4E_cpu();
				}
				else{
					acc4D_cpu();
				}
			}
			else{
				if(Nomp == 1){
					acc4Ef_cpu();
				}
				else{
					acc4Df_cpu();
				}
			}
			if(P.SERIAL_GROUPING == 1){
				Sortb_cpu (Encpairs2_h, 0, N_h[0], P.NencMax);
			}
			kick32Ab_cpu (x4_h, v4_h, a_h, ab_h, rcritv_h, dt_h[0] * Kt[SIn - 1] * def_ksq, Nencpairs_h, Encpairs2_h, 0, N_h[0], P.NencMax, 1);

		}
	}
	if(ForceFlag > 0 || P.setElements > 1){
		comCall(1);
		if(P.setElements > 1){
			setElements_cpu(x4_h, v4_h, index_h, setElementsData_h, setElementsLine_h, Msun_h, dt_h, time_h, N_h[0], Nst, 1);
		}
		if(P.Usegas == 1) GasAccCall(time_h, dt_h, Kt[SIn - 1]);
		if(P.UseGR > 0 || P.UseTides > 0 || P.UseRotationalDeformation > 0 || P.UseJ2 > 0){
			force_cpu (x4_h, v4_h, index_h, spin_h, love_h, Msun_h, Spinsun_h, Lovesun_h, J2_h, vold_h, dt_h, Kt[SIn - 1], time_h, N_h[0], Nst, P.UseGR, P.UseTides, P.UseRotationalDeformation, 0, 1);
		}
		if(P.UseMigrationForce > 0){
			artificialMigration_cpu (x4_h, v4_h, index_h, migration_h, Msun_h, dt_h, Kt[SIn - 1], N_h[0], Nst, 0, 1);
			//artificialMigration2_cpu (x4_h, v4_h, index_h, dt_h, Kt[SIn - 1], N_h[0], Nst, 0, 1);
		}
		if(P.UseYarkovsky == 1) CallYarkovsky2_cpu (x4_h, v4_h, spin_h, index_h, Msun_h, dt_h, Kt[SIn - 1], N_h[0], Nst, 0);
		if(P.UseYarkovsky == 2) CallYarkovsky_averaged_cpu (x4_h, v4_h, spin_h, index_h, Msun_h, dt_h, Kt[SIn - 1], N_h[0], Nst, 0);
		if(P.UsePR == 1) PoyntingRobertsonEffect2_cpu (x4_h, v4_h, index_h, dt_h, Kt[SIn - 1], N_h[0], Nst, 0);
		if(P.UsePR == 2) PoyntingRobertsonEffect_averaged_cpu (x4_h, v4_h, index_h, Msun_h, dt_h, Kt[SIn - 1], N_h[0], Nst, 0);
		comCall(-1);
	}
	EjectionFlag2 = 0;

	for(int si = 0; si < SIn; ++si){
		HCCall(Ct[si], 1);
		fg_cpu (x4_h, v4_h, xold_h, vold_h, index_h, dt_h[0] * FGt[si], Msun_h[0].x, N_h[0], aelimits_h, aecount_h, Gridaecount_h, Gridaicount_h, si, P.UseGR);

		if(Nenc_m[0] > 0){
			for(int i = 0; i < def_GMax; ++i){
				Nenc_m[i] = 0;
			}
			Nencpairs2_h[0] = 0;		
		}
//printf("Nencpairs %d\n", Nencpairs_h[0]);
		if(Nencpairs_h[0] > 0){
			encounter_cpu (x4_h, v4_h, xold_h, vold_h, rcrit_h, rcritv_h, dt_h[0] * FGt[si], Nencpairs_h[0], Nencpairs_h, Encpairs_h, Nencpairs2_h, Encpairs2_h, enccount_h, si, N_h[0], time_h[0], P.StopAtEncounter, Ncoll_m, P.MinMass);

			if(P.StopAtEncounter > 0 && Ncoll_m[0] > 0){
				Ncoll_m[0] = 0;
				StopAtEncounterFlag2 = 1;
			}
			
			if(P.SLevels > 1){
				if(Nencpairs2_h[0] > 0){
					double time = time_h[0];
					SEnc(time, 0, 1.0, si, noColl);
				}
			}
			else{
//printf("Nencpairs2 %d\n", Nencpairs2_h[0]);
				if(Nencpairs2_h[0] > 0){
					group_cpu (Nenc_m, Nencpairs2_h, Encpairs2_h, Encpairs_h, P.NencMax, N_h[0], N_h[0], P.SERIAL_GROUPING);
				}
				BSCall(si, time_h[0], noColl, 1.0);
			}
		}

		if(StopAtEncounterFlag2 == 1){
			StopAtEncounterFlag2 = 0;
			int enc = StopAtEncounterCall();
			if(enc == 0) return 0;
		}
		if(Ncoll_m[0] > 0){
			int col = CollisionCall(noColl);
			if(col == 0) return 0;
		}
		if(CollisionFlag == 1 && P.ei > 0 && timeStep % P.ei == 0){
			int rem = RemoveCall();
			if( rem == 0) return 0;
		}
		if(NWriteEnc_m[0] > 0){
			int enc = writeEncCall();
			if(enc == 0) return 0;
		}
		HCCall(Ct[si], -1);
		if(si < SIn - 1){
			if(P.KickFloat == 0){
				if(Nomp == 1){
					acc4E_cpu();
				}
				else{
					acc4D_cpu();
				}
			}
			else{
				if(Nomp == 1){
					acc4Ef_cpu();
				}
				else{
					acc4Df_cpu();
				}
			}
			if(P.SERIAL_GROUPING == 1){
				Sortb_cpu (Encpairs2_h, 0, N_h[0], P.NencMax);
			}
			kick32Ab_cpu (x4_h, v4_h, a_h, ab_h, rcritv_h, dt_h[0] * Kt[si] * def_ksq, Nencpairs_h, Encpairs2_h, 0, N_h[0], P.NencMax, 1);

			if(ForceFlag > 0){
				comCall(1);
				if(P.Usegas == 1) GasAccCall(time_h, dt_h, Kt[si]);
				if(P.UseGR > 0 || P.UseTides > 0 || P.UseRotationalDeformation > 0 || P.UseJ2 > 0){
					force_cpu (x4_h, v4_h, index_h, spin_h, love_h, Msun_h, Spinsun_h, Lovesun_h, J2_h, vold_h, dt_h, Kt[si], time_h, N_h[0], Nst, P.UseGR, P.UseTides, P.UseRotationalDeformation, 0, 1);
				}
				if(P.UseMigrationForce > 0){
					artificialMigration_cpu (x4_h, v4_h, index_h, migration_h, Msun_h, dt_h, Kt[si], N_h[0], Nst, 0, 1);
					//artificialMigration2_cpu (x4_h, v4_h, index_h, dt_h, Kt[si], N_h[0], Nst, 0, 1);
				}

				if(P.UseYarkovsky == 1) CallYarkovsky2_cpu (x4_h, v4_h, spin_h, index_h, Msun_h, dt_h, Kt[si], N_h[0], Nst, 0);
				if(P.UseYarkovsky == 2) CallYarkovsky_averaged_cpu (x4_h, v4_h, spin_h, index_h, Msun_h, dt_h, Kt[si], N_h[0], Nst, 0);
				if(P.UsePR == 1) PoyntingRobertsonEffect2_cpu (x4_h, v4_h, index_h, dt_h, Kt[si], N_h[0], Nst, 0);
				if(P.UsePR == 2) PoyntingRobertsonEffect_averaged_cpu (x4_h, v4_h, index_h, Msun_h, dt_h, Kt[si], N_h[0], Nst, 0);
				comCall(-1);
			}
		}
	}
	if(P.KickFloat == 0){
		if(Nomp == 1){
			acc4E_cpu();
		}
		else{
			acc4D_cpu();
		}
	}
	else{
		if(Nomp == 1){
			acc4Ef_cpu();
		}
		else{
			acc4Df_cpu();
		}
	}
	if(P.SERIAL_GROUPING == 1){
		Sortb_cpu (Encpairs2_h, 0, N_h[0], P.NencMax);
	}
	kick32Ab_cpu (x4_h, v4_h, a_h, ab_h, rcritv_h, dt_h[0] * Kt[SIn - 1] * def_ksq, Nencpairs_h, Encpairs2_h, 0, N_h[0], P.NencMax, 1);

	if(ForceFlag > 0){
		comCall(1);
		if(P.Usegas == 1) GasAccCall(time_h, dt_h, Kt[SIn - 1]);
		if(P.UseGR > 0 || P.UseTides > 0 || P.UseRotationalDeformation > 0 || P.UseJ2 > 0){
			force_cpu (x4_h, v4_h, index_h, spin_h, love_h, Msun_h, Spinsun_h, Lovesun_h, J2_h, vold_h, dt_h, Kt[SIn - 1], time_h, N_h[0], Nst, P.UseGR, P.UseTides, P.UseRotationalDeformation, 0, 1);
		}
		if(P.UseMigrationForce > 0){
			artificialMigration_cpu (x4_h, v4_h, index_h, migration_h, Msun_h, dt_h, Kt[SIn - 1], N_h[0], Nst, 0, 1);
			//artificialMigration2_cpu (x4_h, v4_h, index_h, dt_h, Kt[SIn - 1], N_h[0], Nst, 0, 1);
		}
		if(P.UseYarkovsky == 1) CallYarkovsky2_cpu (x4_h, v4_h, spin_h, index_h, Msun_h, dt_h, Kt[SIn - 1], N_h[0], Nst, 0);
		if(P.UseYarkovsky == 2) CallYarkovsky_averaged_cpu (x4_h, v4_h, spin_h, index_h, Msun_h, dt_h, Kt[SIn - 1], N_h[0], Nst, 0);
		if(P.UsePR == 1) PoyntingRobertsonEffect2_cpu (x4_h, v4_h, index_h, dt_h, Kt[SIn - 1], N_h[0], Nst, 0);
		if(P.UsePR == 2) PoyntingRobertsonEffect_averaged_cpu (x4_h, v4_h, index_h, Msun_h, dt_h, Kt[SIn - 1], N_h[0], Nst, 0);
		comCall(-1);
	}
	if(EjectionFlag_m[0] > 0){
		int Ej = EjectionCall();
		if(Ej == 0) return 0;
	}

 #if def_poincareFlag == 1
	int per = PoincareSectionCall(time_h[0]);
	if(per == 0) return 0;
 #endif
	return 1;
}

// *************************************************
// Step small CPU
// *************************************************
int Data::step_small_cpu(int noColl){
	Rcrit_cpu (x4_h, v4_h, x4b_h, v4b_h, spin_h, spinb_h, 1.0 / (3.0 * Msun_h[0].x), rcrit_h, rcritb_h, rcritv_h, rcritvb_h, index_h, indexb_h, dt_h[0], n1_h[0], n2_h[0], time_h, time_h[0], EjectionFlag_m, N_h[0] + Nsmall_h[0], NconstT, P.SLevels, noColl);

	//use last time step information for setElements function, the beginning of the time step
	if(P.setElementsV == 2){ // convert barycentric velocities to heliocentric
		comCall(1);
	}
	if(P.setElementsV > 0){
		setElements_cpu(x4_h, v4_h, index_h, setElementsData_h, setElementsLine_h, Msun_h, dt_h, time_h, N_h[0], Nst, 0);
	}
	if(P.setElementsV == 2){
		comCall(-1);
	}

	if(P.SERIAL_GROUPING == 1){
		Sortb_cpu(Encpairs2_h, 0, N_h[0] + Nsmall_h[0], P.NencMax);
	}
	if(EjectionFlag2 == 0){
		kick32Ab_cpu(x4_h, v4_h, a_h, ab_h, rcritv_h, dt_h[0] * Kt[SIn - 1] * def_ksq, Nencpairs_h, Encpairs2_h, 0, N_h[0] + Nsmall_h[0], P.NencMax, 1);
	}
	else{
//
		if(P.KickFloat == 0){
			if(Nomp == 1){
				acc4E_cpu();
				acc4Esmall_cpu();
			}
			else{
				acc4D_cpu();
				acc4Dsmall_cpu();
			}
		}
		else{
			if(Nomp == 1){
				acc4Ef_cpu();
				acc4Efsmall_cpu();
			}
			else{
				acc4Df_cpu();
				acc4Dfsmall_cpu();
			}
		}
		if(P.SERIAL_GROUPING == 1){
			Sortb_cpu (Encpairs2_h, 0, N_h[0] + Nsmall_h[0], P.NencMax);
		}
		kick32Ab_cpu (x4_h, v4_h, a_h, ab_h, rcritv_h, dt_h[0] * Kt[SIn - 1] * def_ksq, Nencpairs_h, Encpairs2_h, 0, N_h[0] + Nsmall_h[0], P.NencMax, 1);

	}
//
	if(ForceFlag > 0 || P.setElements > 1){
		comCall(1);
		if(P.setElements > 1){
			setElements_cpu(x4_h, v4_h, index_h, setElementsData_h, setElementsLine_h, Msun_h, dt_h, time_h, N_h[0], Nst, 1);
		}
		if(P.Usegas == 1){
			GasAccCall(time_h, dt_h, Kt[SIn - 1]);
			GasAccCall_small(time_h, dt_h, Kt[SIn - 1]);
		}
		if(P.Usegas == 2){
			//GasAccCall(time_h, dt_h, Kt[SIn - 1]);
			GasAccCall2_small(time_h, dt_h, Kt[SIn - 1]);
		}
		if(P.UseGR > 0 || P.UseTides > 0 || P.UseRotationalDeformation > 0 || P.UseJ2 > 0){
			force_cpu (x4_h, v4_h, index_h, spin_h, love_h, Msun_h, Spinsun_h, Lovesun_h, J2_h, vold_h, dt_h, Kt[SIn - 1], time_h, N_h[0] + Nsmall_h[0], Nst, P.UseGR, P.UseTides, P.UseRotationalDeformation, 0, 1);
		}
		if(P.UseMigrationForce > 0){
			artificialMigration_cpu (x4_h, v4_h, index_h, migration_h, Msun_h, dt_h, Kt[SIn - 1], N_h[0] + Nsmall_h[0], Nst, 0, 1);
			//artificialMigration2_cpu (x4_h, v4_h, index_h, dt_h, Kt[SIn - 1], N_h[0] + Nsmall_h[0], Nst, 0, 1);
		}
		if(P.UseYarkovsky == 1) CallYarkovsky2_cpu (x4_h, v4_h, spin_h, index_h, Msun_h, dt_h, Kt[SIn - 1], N_h[0] + Nsmall_h[0], Nst, 0);
		if(P.UseYarkovsky == 2) CallYarkovsky_averaged_cpu (x4_h, v4_h, spin_h, index_h, Msun_h, dt_h, Kt[SIn - 1], N_h[0] + Nsmall_h[0], Nst, 0);
		if(P.UsePR == 1) PoyntingRobertsonEffect2_cpu (x4_h, v4_h, index_h, dt_h, Kt[SIn - 1], N_h[0] + Nsmall_h[0], Nst, 0);
		if(P.UsePR == 2) PoyntingRobertsonEffect_averaged_cpu (x4_h, v4_h, index_h, Msun_h, dt_h, Kt[SIn - 1], N_h[0] + Nsmall_h[0], Nst, 0);
		comCall(-1);
	}
	EjectionFlag2 = 0;

	for(int si = 0; si < SIn; ++si){

		HCCall(Ct[si], 1);

		fg_cpu (x4_h, v4_h, xold_h, vold_h, index_h, dt_h[0] * FGt[si], Msun_h[0].x, N_h[0] + Nsmall_h[0], aelimits_h, aecount_h, Gridaecount_h, Gridaicount_h, si, P.UseGR);

		if(P.WriteEncounters == 2 && si == 0){
			Nencpairs2_h[0] = 0;		
			if(UseBVH == 1){
				BVHCall1();
			}
			if(UseBVH == 2){
				BVHCall2();
			}

			if(Nencpairs2_h[0] > 0){
				encounter_small_cpu (x4_h, v4_h, xold_h, vold_h, index_h, spin_h, dt_h[0] * FGt[si], Nencpairs2_h[0], Encpairs2_h, NWriteEnc_m, writeEnc_h, time_h[0]);
			}

			Nencpairs2_h[0] = 0;		
		}

		if(Nenc_m[0] > 0){
			for(int i = 0; i < def_GMax; ++i){
				Nenc_m[i] = 0;
			}
			Nencpairs2_h[0] = 0;		
		}
//printf("Nencpairs %d\n", Nencpairs_h[0]);
		if(Nencpairs_h[0] > 0){
			encounter_cpu (x4_h, v4_h, xold_h, vold_h, rcrit_h, rcritv_h, dt_h[0] * FGt[si], Nencpairs_h[0], Nencpairs_h, Encpairs_h, Nencpairs2_h, Encpairs2_h, enccount_h, si, N_h[0] + Nsmall_h[0], time_h[0], P.StopAtEncounter, Ncoll_m, P.MinMass);

			if(P.StopAtEncounter > 0 && Ncoll_m[0] > 0){
				Ncoll_m[0] = 0;
				StopAtEncounterFlag2 = 1;
			}
			
			if(P.SLevels > 1){
				if(Nencpairs2_h[0] > 0){
					double time = time_h[0];
					SEnc(time, 0, 1.0, si, noColl);
				}
			}
			else{
//printf("Nencpairs2 %d\n", Nencpairs2_h[0]);
				if(Nencpairs2_h[0] > 0){
					group_cpu (Nenc_m, Nencpairs2_h, Encpairs2_h, Encpairs_h, P.NencMax, N_h[0] + Nsmall_h[0], N_h[0], P.SERIAL_GROUPING);
				}
				BSCall(si, time_h[0], noColl, 1.0);
			}
		}

		if(StopAtEncounterFlag2 == 1){
			StopAtEncounterFlag2 = 0;
			int enc = StopAtEncounterCall();
			if(enc == 0) return 0;
		}
		if(Ncoll_m[0] > 0){
			int col = CollisionCall(noColl);
			if(col == 0) return 0;
		}
		if(P.UseSmallCollisions == 1 || P.UseSmallCollisions == 3){
			fragmentCall();
			if(nFragments_m[0] > 0){
				int er = printFragments(nFragments_m[0]);
				if(er == 0) return 0;
				er = RemoveCall();
				if(er == 0) return 0;
			}
		}
		if(P.UseSmallCollisions == 1 || P.UseSmallCollisions == 2){
			rotationCall();
			if(nFragments_m[0] > 0){
				int er = printRotation();
				if(er == 0) return 0;
			}
		}
		if(CollisionFlag == 1 && P.ei > 0 && timeStep % P.ei == 0){
			int rem = RemoveCall();
			if( rem == 0) return 0;
		}
		if(NWriteEnc_m[0] > 0){
			int enc = writeEncCall();
			if(enc == 0) return 0;
		}

		if(P.CreateParticles > 0){
			int er = createCall();
			if(er == 0) return 0;
			if(nFragments_m[0] > 0){
				printCreateparticle(nFragments_m[0]);
			}
		}

		HCCall(Ct[si], -1);
		if(si < SIn - 1){
			if(P.KickFloat == 0){
				if(Nomp == 1){
					acc4E_cpu();
					acc4Esmall_cpu();
				}
				else{
					acc4D_cpu();
					acc4Dsmall_cpu();
				}
			}
			else{
				if(Nomp == 1){
					acc4Ef_cpu();
					acc4Efsmall_cpu();
				}
				else{
					acc4Df_cpu();
					acc4Dfsmall_cpu();
				}
			}
			if(P.SERIAL_GROUPING == 1){
				Sortb_cpu(Encpairs2_h, 0, N_h[0] + Nsmall_h[0], P.NencMax);
			}
			kick32Ab_cpu(x4_h, v4_h, a_h, ab_h, rcritv_h, dt_h[0] * Kt[si] * def_ksq, Nencpairs_h, Encpairs2_h, 0, N_h[0] + Nsmall_h[0], P.NencMax, 1);

			if(ForceFlag > 0){
				comCall(1);
				if(P.Usegas == 1){
					GasAccCall(time_h, dt_h, Kt[si]);
					GasAccCall_small(time_h, dt_h, Kt[si]);
				}
				if(P.Usegas == 2){
					//GasAccCall(time_h, dt_h, Kt[SIn - 1]);
					GasAccCall2_small(time_h, dt_h, Kt[si]);
				}
				if(P.UseGR > 0 || P.UseTides > 0 || P.UseRotationalDeformation > 0 || P.UseJ2 > 0){
					force_cpu (x4_h, v4_h, index_h, spin_h, love_h, Msun_h, Spinsun_h, Lovesun_h, J2_h, vold_h, dt_h, Kt[si], time_h, N_h[0] + Nsmall_h[0], Nst, P.UseGR, P.UseTides, P.UseRotationalDeformation, 0, 1);
				}
				if(P.UseMigrationForce > 0){
					artificialMigration_cpu (x4_h, v4_h, index_h, migration_h, Msun_h, dt_h, Kt[si], N_h[0] + Nsmall_h[0], Nst, 0, 1);
					//artificialMigration2_cpu (x4_h, v4_h, index_h, dt_h, Kt[si], N_h[0] + Nsmall_h[0], Nst, 0, 1);
				}

				if(P.UseYarkovsky == 1) CallYarkovsky2_cpu (x4_h, v4_h, spin_h, index_h, Msun_h, dt_h, Kt[si], N_h[0] + Nsmall_h[0], Nst, 0);
				if(P.UseYarkovsky == 2) CallYarkovsky_averaged_cpu (x4_h, v4_h, spin_h, index_h, Msun_h, dt_h, Kt[si], N_h[0] + Nsmall_h[0], Nst, 0);
				if(P.UsePR == 1) PoyntingRobertsonEffect2_cpu (x4_h, v4_h, index_h, dt_h, Kt[si], N_h[0] + Nsmall_h[0], Nst, 0);
				if(P.UsePR == 2) PoyntingRobertsonEffect_averaged_cpu (x4_h, v4_h, index_h, Msun_h, dt_h, Kt[si], N_h[0] + Nsmall_h[0], Nst, 0);
				comCall(-1);
			}
		}
	}
	if(P.KickFloat == 0){
		if(Nomp == 1){
			acc4E_cpu();
			acc4Esmall_cpu();
		}
		else{
			acc4D_cpu();
			acc4Dsmall_cpu();
		}
	}
	else{
		if(Nomp == 1){
			acc4Ef_cpu();
			acc4Efsmall_cpu();
		}
		else{
			acc4Df_cpu();
			acc4Dfsmall_cpu();
		}
	}

	if(P.SERIAL_GROUPING == 1){
		Sortb_cpu (Encpairs2_h, 0, N_h[0] + Nsmall_h[0], P.NencMax);
	}
	kick32Ab_cpu (x4_h, v4_h, a_h, ab_h, rcritv_h, dt_h[0] * Kt[SIn - 1] * def_ksq, Nencpairs_h, Encpairs2_h, 0, N_h[0] + Nsmall_h[0], P.NencMax, 1);

	
	if(ForceFlag > 0){
		comCall(1);
		if(P.Usegas == 1){
			GasAccCall(time_h, dt_h, Kt[SIn - 1]);
			GasAccCall_small(time_h, dt_h, Kt[SIn - 1]);
		}
		if(P.Usegas == 2){
			//GasAccCall(time_h, dt_h, Kt[SIn - 1]);
			GasAccCall2_small(time_h, dt_h, Kt[SIn - 1]);
		}
		if(P.UseGR > 0 || P.UseTides > 0 || P.UseRotationalDeformation > 0 || P.UseJ2 > 0){
			force_cpu (x4_h, v4_h, index_h, spin_h, love_h, Msun_h, Spinsun_h, Lovesun_h, J2_h, vold_h, dt_h, Kt[SIn - 1], time_h, N_h[0] + Nsmall_h[0], Nst, P.UseGR, P.UseTides, P.UseRotationalDeformation, 0, 1);
		}
		if(P.UseMigrationForce > 0){
			artificialMigration_cpu (x4_h, v4_h, index_h, migration_h, Msun_h, dt_h, Kt[SIn - 1], N_h[0] + Nsmall_h[0], Nst, 0, 1);
			//artificialMigration2_cpu (x4_h, v4_h, index_h, dt_h, Kt[SIn - 1], N_h[0] + Nsmall_h[0], Nst, 0, 1);
		}
		if(P.UseYarkovsky == 1) CallYarkovsky2_cpu (x4_h, v4_h, spin_h, index_h, Msun_h, dt_h, Kt[SIn - 1], N_h[0] + Nsmall_h[0], Nst, 0);
		if(P.UseYarkovsky == 2) CallYarkovsky_averaged_cpu (x4_h, v4_h, spin_h, index_h, Msun_h, dt_h, Kt[SIn - 1], N_h[0] + Nsmall_h[0], Nst, 0);
		if(P.UsePR == 1) PoyntingRobertsonEffect2_cpu (x4_h, v4_h, index_h, dt_h, Kt[SIn - 1], N_h[0] + Nsmall_h[0], Nst, 0);
		if(P.UsePR == 2) PoyntingRobertsonEffect_averaged_cpu (x4_h, v4_h, index_h, Msun_h, dt_h, Kt[SIn - 1], N_h[0] + Nsmall_h[0], Nst, 0);
		comCall(-1);
	}
	if(EjectionFlag_m[0] > 0){
		int Ej = EjectionCall();
		if(Ej == 0) return 0;
	}

 #if def_poincareFlag == 1
	int per = PoincareSectionCall(time_h[0]);
	if(per == 0) return 0;
 #endif
	return 1;
}
#endif



// *************************************************
// Step large N
// *************************************************
// *************************************************
// Step small
// *************************************************

// *************************************************
// Step M
// *************************************************
// *************************************************
// Step M3
// *************************************************
// *************************************************
// Step M simple
// *************************************************

#if def_TTV == 2
#endif
