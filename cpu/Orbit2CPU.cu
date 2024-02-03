#include "Orbit2CPU.h"

//Constructor
 Data::Data(long long Restart): Host(Restart){


}

//Allocate orbit data
 int Data::AllocateOrbit(){
	int error;

	//allocate memory on host//
	rcrit_h = (double*)malloc(NconstT * P.SLevels * sizeof(double));
	x4_h = (double4*)malloc(NconstT * sizeof(double4));
	v4_h = (double4*)malloc(NconstT * sizeof(double4));
	v4Helio_h = (double4*)malloc(NconstT * sizeof(double4));
	index_h = (int*)malloc(NconstT * sizeof(int));
	spin_h = (double4*)malloc(NconstT * sizeof(double4));
	love_h = (double3*)malloc(NconstT * sizeof(double3));
	if(P.UseMigrationForce > 0){
		migration_h = (double3*)malloc(NconstT * sizeof(double3));
	}
	else{
		migration_h = NULL;
	}
	if(P.CreateParticles > 0){
		createFlag_h = (int*)malloc(NconstT * sizeof(int));
	}
	else{
		createFlag_h = NULL;
	}
	U_h = (double*)malloc(Nst * sizeof(double));
	LI_h = (double*)malloc(Nst * sizeof(double));
	LI0_h = (double*)malloc(Nst * sizeof(double));
	Energy_h = (double*)malloc(NEnergyT * sizeof(double));
	Energy0_h = (double*)malloc(Nst * sizeof(double));
	Coll_h = (double*)malloc(def_NColl * def_MaxColl * Nst * sizeof(double));
	writeEnc_h = (double*)malloc(def_NColl * def_MaxWriteEnc * Nst * sizeof(double));
	Fragments_h = (double*)malloc(25 * P.Nfragments * Nst * sizeof(double));
	aelimits_h = (float4*)malloc(NconstT * sizeof(float4));
	aecount_h = (unsigned int*)malloc(NconstT * sizeof(unsigned int));
	enccount_h = (unsigned int*)malloc(NconstT * sizeof(unsigned int));
	aecountT_h = (unsigned long long*)malloc(NconstT * sizeof(unsigned long long));
	enccountT_h = (unsigned long long*)malloc(NconstT * sizeof(unsigned long long));

	coordinateBuffer_h = (double*)malloc(P.Buffer * def_BufferSize * NconstT * sizeof(double));
	coordinateBufferIrr_h = (double*)malloc(P.Buffer * def_BufferSize * NconstT * sizeof(double));
	timestepBuffer = (long long int*)malloc(P.Buffer * sizeof(long long int));
	timestepBufferIrr = (long long int*)malloc(P.Buffer * sizeof(long long int));
	NBuffer = (int2*)malloc(Nst * P.Buffer * sizeof(int2));
	NBufferIrr = (int2*)malloc(Nst * P.Buffer * sizeof(int2));
	



#if def_TTV > 0
	{
		int n = def_NtransitTimeMax * NconstT;
		if(def_TTV == 2 && P.PrintTransits == 0){
			n = 0;
		}
		TransitTime_h = (double*)malloc(n * sizeof(double));
		TransitTimeObs_h = (double2*)malloc(def_NtransitTimeMax * N_h[0] * sizeof(double2));
		NtransitsT_h = (int2*)malloc(NconstT * sizeof(int2));
		NtransitsTObs_h = (int*)malloc(N_h[0] * sizeof(int));
	}
#else
	TransitTime_h = NULL;
	TransitTimeObs_h = NULL;
	NtransitsT_h = NULL;
	NtransitsTObs_h = NULL;
#endif

#if def_RV > 0
	RV_h = (double2*)malloc(def_NRVMax * Nst * sizeof(double2));
	RVObs_h = (double3*)malloc(def_NRVMax * Nst * sizeof(double3));
	NRVT_h = (int2*)malloc(Nst * sizeof(int2));
	NRVTObs_h = (int*)malloc(Nst * sizeof(int));
#else
	RV_h = NULL;
	RVObs_h = NULL;
	NRVT_h = NULL;
	NRVTObs_h = NULL;
#endif

#if def_TTV > 0
	elementsA_h = (double4*)malloc(NconstT * sizeof(double4));
	elementsB_h = (double4*)malloc(NconstT * sizeof(double4));
	elementsT_h = (double4*)malloc(NconstT * sizeof(double4));
	elementsSpin_h = (double4*)malloc(NconstT * sizeof(double4));
	elementsL_h = (elements10*)malloc(NconstT * sizeof(elements10));
	elementsC_h = (int2*)malloc((Nst + MCMC_NT) * sizeof(int2));
	elementsP_h = (double4*)malloc(Nst * sizeof(double4));
	elementsSA_h = (double*)malloc(Nst * sizeof(double));
	elementsI_h = (int4*)malloc(NconstT * sizeof(int4));
	elementsM_h = (double*)malloc(Nst * sizeof(double));

  #if MCMC_NCOV > 0
	elementsCOV_h = (double*)malloc(NconstT * N_h[0] * MCMC_NCOV * MCMC_NCOV * sizeof(double));
  #else
	elementsCOV_h = NULL;
  #endif
#else
	elementsA_h = NULL;
	elementsB_h = NULL;
	elementsT_h = NULL;
	elementsSpin_h = NULL;
	elementsL_h = NULL;
	elementsC_h = NULL;
	elementsP_h = NULL;
	elementsSA_h = NULL;
	elementsI_h = NULL;
	elementsM_h = NULL;
	elementsCOV_h = NULL;
#endif

	groupIterate_h = (int*)malloc(sizeof(int));

#if def_poincareFlag == 1
	PFlag_h = (int*)malloc(sizeof(int));
	PFlag_h[0] = 0;
#endif

	BSAstop_h = (int*)malloc(sizeof(int));

	error = 0;
	fprintf(masterfile,"CPU malloc error = %d = %s\n",error, "-");
	if(error != 0){
		printf("CPU malloc error = %d = %s\n",error, "-");
		return 0;
	}

	//allocate pinned memory on host//
#if def_CPU == 0
	cudaHostAlloc((void **)&test_h, NconstT * sizeof(double), cudaHostAllocDefault);
	cudaHostAlloc((void **)&Nencpairs_h, P.ndev * (Nst + 1) * sizeof(int), cudaHostAllocDefault);
	cudaHostAlloc((void **)&Nencpairs2_h, (Nst + 1) * sizeof(int), cudaHostAllocDefault);
	cudaHostAlloc((void **)&Nencpairs3_h, P.SLevels * sizeof(int), cudaHostAllocDefault);
#else
	test_h = (double*)malloc(NconstT * sizeof(double));
	Nencpairs_h = (int*)malloc(P.ndev * (Nst + 1) * sizeof(int));
	Nencpairs2_h = (int*)malloc((Nst + 1) * sizeof(int));
	Nencpairs3_h = (int*)malloc(P.SLevels * sizeof(int));

#endif

	error = 0;
	fprintf(masterfile,"CPU HostAlloc error = %d = %s\n",error, "-");
	if(error != 0){
		printf("CPU HostAlloc error = %d = %s\n",error, "-");
		return 0;
	}

	//allocate memory on device//
	if(P.UseMigrationForce > 0){
	}
	else{
		migration_h = NULL;
	}
	if(P.CreateParticles > 0){
	}
	else{
		createFlag_h = NULL;
	}


	if(P.WriteEncounters == 2){
	}
	else{
		morton_h = nullptr;
		sortRank_h = nullptr;
		sortCount_h = nullptr;
		sortIndex_h = nullptr;
		leafNodes_h = nullptr;
		internalNodes_h = nullptr;
	}

	if(Nst > 1){
	}
	else{
		groupIndex_h = NULL;
	}


#if def_CPU == 1
	//arrays for backup step
	xold_h = (double4*)malloc(NconstT * sizeof(double4));
	vold_h = (double4*)malloc(NconstT * sizeof(double4));
	a_h = (double3*)malloc(NconstT * sizeof(double3));
	b_h = (double3*)malloc(Nomp * NconstT * sizeof(double3));


	x4b_h = (double4*)malloc(NconstT * sizeof(double4));
	v4b_h = (double4*)malloc(NconstT * sizeof(double4));
	x4bb_h = (double4*)malloc(NconstT * sizeof(double4));
	v4bb_h = (double4*)malloc(NconstT * sizeof(double4));
	ab_h = (double3*)malloc(NconstT * sizeof(double3));
	indexb_h = (int*)malloc(NconstT * sizeof(int));
	indexbb_h = (int*)malloc(NconstT * sizeof(int));

	rcritb_h = (double*)malloc(NconstT * P.SLevels * sizeof(double));
	rcritbb_h = (double*)malloc(NconstT * P.SLevels * sizeof(double));
	rcritv_h = (double*)malloc(NconstT * P.SLevels * sizeof(double));
	rcritvb_h = (double*)malloc(NconstT * P.SLevels * sizeof(double));
	rcritvbb_h = (double*)malloc(NconstT * P.SLevels * sizeof(double));

	spinb_h = (double4*)malloc(NconstT * sizeof(double4));
	spinbb_h = (double4*)malloc(NconstT * sizeof(double4));

	vcom_h = (double3*)malloc(Nst * sizeof(double3));
	EnergySum_h = (double*)malloc(NconstT * sizeof(double));
	Encpairs_h = (int2*)malloc(NBNencT * sizeof(int2));
	Encpairs2_h = (int2*)malloc(NBNencT * sizeof(int2));
	Encpairs3_h = (int*)malloc(NBNencT * P.SLevels * sizeof(int));
	scan_h = (int2*)malloc(NconstT * sizeof(int2));

	if(Nst > 1){
		groupIndex_h = (int2*)malloc(NconstT * sizeof(int2));
	}
	else{
		groupIndex_h = NULL;
	}

	xt_h = (double4*)malloc(NconstT * sizeof(double4));
	vt_h = (double4*)malloc(NconstT * sizeof(double4));
	xp_h = (double4*)malloc(NconstT * sizeof(double4));
	vp_h = (double4*)malloc(NconstT * sizeof(double4));
	dx_h = (double3*)malloc(NconstT * 8 * sizeof(double3));
	dv_h = (double3*)malloc(NconstT * 8 * sizeof(double3));
	dt1_h = (double*)malloc(NconstT * sizeof(double));
	t1_h = (double*)malloc(NconstT * sizeof(double));
	dtgr_h = (double*)malloc(NconstT * sizeof(double));
	Coltime_h = (double*)malloc(sizeof(double));
	BSstop_h = (int*)malloc(sizeof(int));
	BSAstop_h = (int*)malloc(sizeof(int));

  #if USE_RANDOM == 1
	srand48(time(NULL));
	random_h = (int*)malloc(NconstT * sizeof(int));
  #else
	random_h = NULL;
  #endif
	if(P.WriteEncounters == 2){
		morton_h = (unsigned int*)malloc(NconstT * sizeof(unsigned int));
		sortRank_h = (unsigned int*)malloc(NconstT * sizeof(unsigned int));
		sortCount_h = (unsigned int*)malloc(((NconstT + 255) / 256 + 1) * 16 * sizeof(unsigned int));
		sortIndex_h = (int2*)malloc(NconstT * sizeof(int2));
		leafNodes_h = (Node*)malloc(NconstT * sizeof(Node));
		internalNodes_h = (Node*)malloc(NconstT * sizeof(Node));
	}
	else{
		morton_h = nullptr;
		sortRank_h = nullptr;
		sortCount_h = nullptr;
		sortIndex_h = nullptr;
		leafNodes_h = nullptr;
		internalNodes_h = nullptr;
	}

#endif

	// ------------------------
	// MultiGPU allocation
#if def_CPU == 0
	if(P.ndev > 1){
	}
	if(P.ndev > 2){
	}
	if(P.ndev > 3){
	}
	if(P.ndev < 2){
		rcritv_h1 = nullptr;
		x4_h1 = nullptr;
		Nencpairs_h1 = nullptr;
		Encpairs_h1 = nullptr;
		Encpairs2_h1 = nullptr;
	}
	if(P.ndev < 3){
		rcritv_h2 = nullptr;
		x4_h2 = nullptr;
		Nencpairs_h2 = nullptr;
		Encpairs_h2 = nullptr;
		Encpairs2_h2 = nullptr;
	}
	if(P.ndev < 4){
		rcritv_h3 = nullptr;
		x4_h3 = nullptr;
		Nencpairs_h3 = nullptr;
		Encpairs_h3 = nullptr;
		Encpairs2_h3 = nullptr;
	}

	if(P.ndev > 1){
	}
#endif
	// ------------------------


#if def_TTV == 1
#else
	Transit_d = NULL;

#endif

#if def_TTV > 0
	{
		int n = def_NtransitTimeMax * NconstT;
		if(def_TTV == 2 && P.PrintTransits == 0){
			n = 0;
		}
	}
#else
	TransitTime_d = NULL;
	TransitTimeObs_d = NULL;
	NtransitsT_d = NULL;
	NtransitsTObs_d = NULL;
#endif

#if def_RV > 0
#else
	RV_d = NULL;
	RVObs_d = NULL;
	NRVT_d = NULL;
	NRVTObs_d = NULL;
	RVP_d = NULL;
#endif
#if def_TTV > 0
  #if MCMC_BLOCK == 5
  #else
	elementsG_d = NULL;
	elementsGh_d = NULL;
	elementsD_d = NULL;
	elementsMean_d = NULL;
	elementsVar_d = NULL;
  #endif
  #if MCMC_BLOCK == 6
printf("size %lu %lu %lu\n", sizeof(double), sizeof(elements), Nst * (N_h[0] + 1) * sizeof(elements));
  #else
	Symplex_d = NULL;
	SymplexCount_d = NULL;
  #endif
  #if MCMC_BLOCK == 7
  #else
	elementsStep_d = NULL;
	elementsHist_d = NULL;
  #endif

  #if MCMC_NCOV > 0
  #else
	elementsCOV_d = NULL;
  #endif
#else
	elementsA_d = NULL;
	elementsB_d = NULL;
	elementsT_d = NULL;
	elementsSpin_d = NULL;
	elementsAOld_d = NULL;
	elementsAOld2_d = NULL;
	elementsBOld_d = NULL;
	elementsBOld2_d = NULL;
	elementsTOld_d = NULL;
	elementsTOld2_d = NULL;
	elementsSpinOld_d = NULL;
	elementsSpinOld2_d = NULL;
	elementsL_d = NULL;
	elementsC_d = NULL;
	elementsP_d = NULL;
	elementsSA_d = NULL;
	elementsI_d = NULL;
	elementsM_d = NULL;
	elementsG_d = NULL;
	elementsGh_d = NULL;
	elementsD_d = NULL;
	elementsMean_d = NULL;
	elementsVar_d = NULL;
	elementsCOV_d = NULL;
	Symplex_d = NULL;
	SymplexCount_d = NULL;
	elementsStep_d = NULL;
	elementsHist_d = NULL;
#endif

#if def_TTV == 2
#else
	timeold_d = NULL;
	lastTransitTime_d = NULL;
	transitIndex_d = NULL;
	EpochCount_d = NULL;
	TTV_d = NULL;

#endif
	//arrays for backup step


	//arrays for BSA
#if def_G3 > 0
#else
	K_h = NULL;
	Kold_h = NULL;
	x4G3_d = NULL;
	v4G3_d = NULL;
	
#endif

#if def_poincareFlag == 1
#endif

#if USE_RANDOM == 1
#else
	random_h = NULL;
#endif

	CollisionFlag = 0;

	error = 0;
	if(error != 0){
		return 0;
	}

	return 1;
};


//This function allocates mapped memory
 int Data::CMallocateOrbit(){
#if def_CPU == 0
	int error;

	cudaHostAlloc((void **)&Nenc_m, def_GMax * sizeof(int), cudaHostAllocMapped);
	cudaHostGetDevicePointer((void **)&Nenc_m, (void *)Nenc_m, 0);

	cudaHostAlloc((void **)&Ncoll_m, sizeof(int), cudaHostAllocMapped);
	cudaHostGetDevicePointer((void **)&Ncoll_m, (void *)Ncoll_m, 0);

	cudaHostAlloc((void **)&Ntransit_m, sizeof(int), cudaHostAllocMapped);
	cudaHostGetDevicePointer((void **)&Ntransit_d, (void *)Ntransit_m, 0);

	cudaHostAlloc((void **)&NWriteEnc_m, sizeof(int), cudaHostAllocMapped);
	cudaHostGetDevicePointer((void **)&NWriteEnc_m, (void *)NWriteEnc_m, 0);

	cudaHostAlloc((void **)&EjectionFlag_m, (Nst + 1)*sizeof(int), cudaHostAllocMapped);
	cudaHostGetDevicePointer((void **)&EjectionFlag_m, (void *)EjectionFlag_m, 0);

	cudaHostAlloc((void **)&nFragments_m, sizeof(int), cudaHostAllocMapped);
	cudaHostGetDevicePointer((void **)&nFragments_m, (void *)nFragments_m, 0);

	cudaHostAlloc((void **)&EncFlag_m, sizeof(int), cudaHostAllocMapped);
	cudaHostGetDevicePointer((void **)&EncFlag_m, (void *)EncFlag_m, 0);

	cudaHostAlloc((void **)&StopFlag_m, sizeof(int), cudaHostAllocMapped);
	cudaHostGetDevicePointer((void **)&StopFlag_d, (void *)StopFlag_m, 0);

	cudaHostAlloc((void **)&ErrorFlag_m, sizeof(int), cudaHostAllocMapped);
	cudaHostGetDevicePointer((void **)&ErrorFlag_d, (void *)ErrorFlag_m, 0);
#else
	Nenc_m = (int*)malloc(def_GMax * sizeof(int));
	Ncoll_m = (int*)malloc(sizeof(int));
	Ntransit_m = (int*)malloc(sizeof(int));
	NWriteEnc_m = (int*)malloc(sizeof(int));
	EjectionFlag_m = (int*)malloc((Nst + 1) * sizeof(int));
	nFragments_m = (int*)malloc(sizeof(int));
	EncFlag_m = (int*)malloc(sizeof(int));
	StopFlag_m = (int*)malloc(sizeof(int));
	ErrorFlag_m = (int*)malloc(sizeof(int));
#endif
	EncFlag_m[0] = 0;
	StopFlag_m[0] = 0;
	ErrorFlag_m[0] = 0;

	error = 0;
	fprintf(masterfile,"mapping error = %d = %s\n",error, "-");
	if(error != 0){
		printf("mapping error = %d = %s\n",error, "-");
		 return 0;
	}

	return 1;

}


//This function allocates the Gridae and set values to zero
 int Data::GridaeAlloc(){
	int error;
	GridNae = Gridae.Na * Gridae.Ne;
	Gridaecount_h = (unsigned int*)malloc(GridNae * sizeof(unsigned int));
	GridaecountT_h = (unsigned long long*)malloc(GridNae * sizeof(unsigned long long));
	GridaecountS_h = (unsigned long long*)malloc(GridNae * sizeof(unsigned long long));

	for(int i = 0; i < GridNae; ++i){
		Gridaecount_h[i] = 0u;
		GridaecountT_h[i] = 0ull;
		GridaecountS_h[i] = 0ull;
	}
	GridNai = Gridae.Na * Gridae.Ni;
	Gridaicount_h = (unsigned int*)malloc(GridNai * sizeof(unsigned int));
	GridaicountT_h = (unsigned long long*)malloc(GridNai * sizeof(unsigned long long));
	GridaicountS_h = (unsigned long long*)malloc(GridNai * sizeof(unsigned long long));

	for(int i = 0; i < GridNai; ++i){
		Gridaicount_h[i] = 0u;
		GridaicountT_h[i] = 0ull;
		GridaicountS_h[i] = 0ull;
	}

	constantCopy();

	error = 0;
	fprintf(masterfile,"GrideaeAlloc error = %d = %s\n",error, "-");
	if(error != 0){
		printf("GrideaeAlloc error = %d = %s\n",error, "-");
		return 0;
	}

	return 1;
}


 int Data::FGAlloc(){
	int error;
	double S_h[def_FGN + 1];
	double C_h[def_FGN + 1];

	//Table for fastfg//
	for (int j = 0; j<= def_FGN; ++j) {
		double dEj = j*PI_N;
		S_h[j] = sin(dEj);
		C_h[j] = cos(dEj);
	}
	constantCopySC(S_h, C_h);
	error = 0;
	fprintf(masterfile,"FGAlloc error = %d = %s\n",error, "-");
	if(error != 0){
		printf("FGAlloc error = %d = %s\n",error, "-");
		return 0;
	}
	return 1;
}


//This function reads at a restart the correspondent Gridae file
 int Data::readGridae(){
	if(P.tRestart > 0){
		sprintf(Gridae.filename, "aeCount%s_%.*lld.dat", Gridae.X, def_NFileNameDigits, P.tRestart);
		Gridae.file = fopen(Gridae.filename, "r");
		if(Gridae.file == NULL){
			fprintf(masterfile, "Error: aeGrid file not found: aeCount%s_%.*lld.dat\n", Gridae.X, def_NFileNameDigits, P.tRestart);
			printf("Error: aeGrid file not found: aeCount%s_%.*lld.dat\n", Gridae.X, def_NFileNameDigits, P.tRestart);
			return 0;
		}
		//Read Total aeGrid
		int er = 0;
		for(int i = 0; i < Gridae.Ne; ++i){
			for(int j = 0; j < Gridae.Na; ++j){
				er = fscanf(Gridae.file, "%lld",&GridaecountT_h[i * Gridae.Na + j]);
				if(er <= 0){
					return 0;
				}
			}
		}
		//Skip Temporal aeGrid
		int skip;
		for(int i = 0; i < Gridae.Ne; ++i){
			for(int j = 0; j < Gridae.Na; ++j){
				er = fscanf(Gridae.file, "%d",&skip);
				if(er <= 0){
					return 0;
				}
			}
		}
		//Read Total aiGrid
		for(int i = 0; i < Gridae.Ni; ++i){
			for(int j = 0; j < Gridae.Na; ++j){
				er = fscanf(Gridae.file, "%lld",&GridaicountT_h[i * Gridae.Na + j]);
				if(er <= 0){
					return 0;
				}
			}
		}
		fclose(Gridae.file);
	}
	return 1;
}

//This function copies values from the current Gridae to the total and summing host Grid
 int Data::copyGridae(){
	int error;
	//ae grid
	for(int i = 0; i < Gridae.Ne; ++i){
		for(int j = 0; j < Gridae.Na; ++j){
			if(timeStep > Gridae.Start){
				GridaecountS_h[i * Gridae.Na + j] += Gridaecount_h[i * Gridae.Na + j];
				GridaecountT_h[i * Gridae.Na + j] += Gridaecount_h[i * Gridae.Na + j];
			}
		}
	}
	memset(Gridaecount_h, 0, sizeof(int)*GridNae);
	//ae grid
	for(int i = 0; i < Gridae.Ni; ++i){
		for(int j = 0; j < Gridae.Na; ++j){
			if(timeStep > Gridae.Start){
				GridaicountS_h[i * Gridae.Na + j] += Gridaicount_h[i * Gridae.Na + j];
				GridaicountT_h[i * Gridae.Na + j] += Gridaicount_h[i * Gridae.Na + j];
			}
		}
	}
	memset(Gridaicount_h, 0, sizeof(int)*GridNai);
	error = 0;
	fprintf(masterfile,"Grideae copy error = %d = %s\n",error, "-");
	if(error != 0){
		printf("Grideae copy error = %d = %s\n",error, "-");
		return 0;
	}

	return 1;
}


//This function reads the covariance matrix for MCMC sampling
//The file must contain the Cholesky decompoistion part L from COV = L L^T
 int Data::readMCMC_COV(){
	FILE *COVfile;

	COVfile = fopen("MCMCL.dat", "r");
	if(COVfile == NULL){
		fprintf(masterfile, "Error: MCMCL.dat file not found\n");
		printf("Error: MCMCL.dat file not found\n");
		return 0;
	}
	int ii, jj;
	int er = 0;
	for(int i = 0; i < NconstT * MCMC_NCOV; ++i){
		for(int j = 0; j < N_h[0] * MCMC_NCOV; ++j){
			er = fscanf(COVfile, "%d",&ii);
			er = fscanf(COVfile, "%d",&jj);
			er = fscanf(COVfile, "%lf",&elementsCOV_h[i * N_h[0] * MCMC_NCOV + j]);
//printf("MCMCL %d %d %d %g\n", i, i % N_h[0], j, elementsCOV_h[i * N_h[0] * MCMC_NCOV + j]);
			if(er <= 0){
				return 0;
			}
			int iii = 0;
			int NM = N_h[0] * MCMC_NCOV;
			if(NM != 0) iii = i % NM;
			if(ii != iii || jj != j){
				fprintf(masterfile, "Error: MCMCL.dat file not the correct size %d %d %d %d\n", ii, iii, jj, j);
				printf("Error: MCMCL.dat file not the correct size %d %d %d %d\n", ii, iii, jj, j);
				return 0;
			}
		}
	}
	return 1;
}

void BufferInit_cpu(double *coordinateBuffer_h, const int N){

	int id = 0 * 1 + 0;
	for(id = 0 * 1 + 0; id < N; ++id){
		coordinateBuffer_h[id] = 0.0;
	}
}

#if USE_RANDOM == 1
void randomInit_cpu(int *random_h, const int N){
  #if def_CPU == 0
	int id = 0 * 1 + 0;

	for(id = 0 * 1 + 0; id < N; ++id){
		//curand_init(0, id, 0, &random_h[id]);
		curand_init(clock64(), id, 0, &random_h[id]);
	}
  #endif
}

#endif



//This function initializes the data
 int Data::init(){

	doTransits = 0;
#if def_TTV == 1
	doTransits = 1;
#endif
	Ncoll_m[0] = 0;
	Ntransit_m[0] = 0;
	NWriteEnc_m[0] = 0;
	nFragments_m[0] = 0;
	for(int i = 0; i < def_GMax; ++i){
		Nenc_m[i] = 0;
	}
	EjectionFlag_m[0] = 0;
	for(int i = 0; i < NconstT * P.SLevels; ++i){
		rcrit_h[i] = 0.0;
	}
	for(int i = 0; i < NconstT; ++i){
		index_h[i] = -1;
		x4_h[i].x = 1.0;
		x4_h[i].y = 0.0;
		x4_h[i].z = 0.0; 
		x4_h[i].w = -1.0e-12;
		v4_h[i].x = 0.0;
		v4_h[i].y = 0.0;
		v4_h[i].z = 0.0;
		v4_h[i].w = 0.0;
		test_h[i] = -1.0;
		spin_h[i].x = 0.0;
		spin_h[i].y = 0.0;
		spin_h[i].z = 0.0;
		spin_h[i].w = 0.4;	//2.0/5.0
		love_h[i].x = 0.0;
		love_h[i].y = 0.0;
		love_h[i].z = 0.0;
		if(P.UseMigrationForce > 0){ 
			migration_h[i].x = 0.0;
			migration_h[i].y = 0.0;
			migration_h[i].z = 0.0;
		}
		if(P.CreateParticles > 0){
			createFlag_h[i] = -1;
		}
		aelimits_h[i].x = 0.0f;
		aelimits_h[i].y = 1.0f;
		aelimits_h[i].z = 0.0f;
		aelimits_h[i].w = 1.0f;
		aecount_h[i] = 0u;
		enccount_h[i] = 0u;
		aecountT_h[i] = 0ull;
		enccountT_h[i] = 0ull;
#if def_TTV > 0
		elementsA_h[i].x = 0.0;
		elementsA_h[i].y = 0.0;
		elementsA_h[i].z = 0.0;
		elementsA_h[i].w = -1.0e-12;
		elementsB_h[i].x = 0.0;
		elementsB_h[i].y = 0.0;
		elementsB_h[i].z = 0.0;
		elementsB_h[i].w = 0.0;
		elementsT_h[i].x = 0.0;
		elementsT_h[i].y = 0.0;
		elementsT_h[i].z = 0.0;
		elementsT_h[i].w = 0.0;
		elementsSpin_h[i].x = 0.0;
		elementsSpin_h[i].y = 0.0;
		elementsSpin_h[i].z = 0.0;
		elementsSpin_h[i].w = 0.0;
		elementsL_h[i].P = 0.0;
		elementsL_h[i].T = 0.0;
		elementsL_h[i].m = 0.0;
		elementsL_h[i].e = 0.0;
		elementsL_h[i].w = 0.0;
		elementsL_h[i].inc = 0.0;
		elementsL_h[i].O = 0.0;
		elementsL_h[i].r = 0.0;
		elementsL_h[i].a = 0.0;
		elementsL_h[i].M = 0.0;
		elementsI_h[i].x = 0;
		elementsI_h[i].y = 0;
		elementsI_h[i].z = 0;
		elementsI_h[i].w = P.mcmcNE * N_h[0];


		if(i < Nst){
			elementsP_h[i].x = 1.0e300;		//initial value for sum
			elementsP_h[i].y = 0.0;		//contains later a random number
			elementsP_h[i].z = 1.0e300;	//new p
			elementsP_h[i].w = 1.0;		//tunig factor according to acceptance rate
			elementsSA_h[i] = 1.0;
			elementsM_h[i] = Msun_h[i].x;
		}
		if(i < Nst + MCMC_NT){
			elementsC_h[i].x = 0;
			elementsC_h[i].y = 0;
		}
#endif
	}
#if def_TTV > 0
  #if MCMC_NCOV > 0
	for(int j = 0; j < NconstT * N_h[0] * MCMC_NCOV * MCMC_NCOV; ++j){
		elementsCOV_h[j] = 0.0;
	}
  #endif
#endif
	for(int st = 0; st < Nst; ++st){
		EjectionFlag_m[st + 1] = 0;
		for(int i = 0; i < N_h[st] + Nsmall_h[st]; ++i){
			index_h[NBS_h[st] + i] = i + st * def_MaxIndex;
		}
	}
	for(int i = 0; i < P.Buffer * def_BufferSize * NconstT; ++i){
		coordinateBuffer_h[i] = 0.0;
		coordinateBufferIrr_h[i] = 0.0;
	}
	for(int i = 0; i < P.Buffer; ++i){
		timestepBuffer[i] = 0ll;
		timestepBufferIrr[i] = 0ll;
		for(int st = 0; st < Nst; ++st){
			NBuffer[i * Nst + st].x = N_h[st];
			NBuffer[i * Nst + st].y = Nsmall_h[st];
			NBufferIrr[i * Nst + st].x = N_h[st];
			NBufferIrr[i * Nst + st].y = Nsmall_h[st];
		}
	}
	BufferInit_cpu /* (P.Buffer * def_BufferSize * NconstT + 511) / 512, 512 */ (coordinateBuffer_h, P.Buffer * def_BufferSize * NconstT);
	BufferInit_cpu /* (P.Buffer * def_BufferSize * NconstT + 511) / 512, 512 */ (coordinateBufferIrr_h, P.Buffer * def_BufferSize * NconstT);
	for(int i = 0; i < NEnergyT; ++i){
		Energy_h[i] = 0.0;
	}

	for(int i = 0; i < Nst * def_NColl * def_MaxColl; ++i){
		Coll_h[i] = 0.0;
	}

	for(int i = 0; i < Nst * def_NColl * def_MaxWriteEnc; ++i){
		writeEnc_h[i] = 0.0;
	}

	for(int i = 0; i < Nst * 25 * P.Nfragments; ++i){
		Fragments_h[i] = 0.0;
	}

	for(int st = 0; st < P.ndev * (Nst + 1); ++st){
		Nencpairs_h[st] = 0;
	}
	for(int st = 0; st < Nst + 1; ++st){
		Nencpairs2_h[st] = 0;
	}
	for(int i = 0; i < P.SLevels; ++i){
		Nencpairs3_h[i] = 0;
	}
	for(int st = 0; st < Nst; ++st){
		U_h[st] = 0.0;
		LI_h[st] = 0.0;
		LI0_h[st] = 1.0;
		Energy0_h[st] = 1.0;
	}

#if USE_RANDOM == 1
	randomInit_cpu /* (NconstT + 255) / 256, 256*/ (random_h, NconstT);
#endif

	return 1;
}


//This function calls the readic function and copies the data to the GPU.
 int Data::ic(){
	for(int st = 0; st < Nst; ++st){
		if(N_h[st] + Nsmall_h[st] > 0){
			GSF[st].logfile = fopen(GSF[st].logfilename, "a");
			int NBS = NBS_h[st];
			fprintf(GSF[st].logfile, "\n************* Read initial conditions ****************\n \n");
			int icerr = 0;
			icerr = readic(st);
			if(icerr == 0){
				printf("Error: Could not read initial conditions\n");
				fprintf(GSF[st].logfile, "Error: Could not read initial conditions\n");
				fprintf(masterfile, "Error in Simulation %s\n", GSF[st].path);
				return 0;
			}
			if(Nsmall_h[st] < Nmin[st].y && P.UseTestParticles > 0){
				printf("Error: No Test Particles found\n");
				fprintf(GSF[st].logfile, "Error: No Test Particles found\n");
				fprintf(masterfile, "Error: No Test Particles found %s\n", GSF[st].path);
				return 0;
			}
			fclose(GSF[st].logfile);
			HelioToDemo(x4_h + NBS, v4_h + NBS, Msun_h[st].x, N_h[st] + Nsmall_h[st]);
			//HelioToBary(x4_h + NBS, v4_h + NBS, Msun_h[st].x, N_h[st] + Nsmall_h[st]);
		}
	}
	//Copy memory to device//
	if(P.UseMigrationForce > 0){
	}


#if def_CPU == 1
	memcpy(x4b_h, x4_h, sizeof(double4) * NconstT);
	memcpy(v4b_h, v4_h, sizeof(double4) * NconstT);
	memcpy(x4bb_h, x4_h, sizeof(double4) * NconstT);
	memcpy(v4bb_h, v4_h, sizeof(double4) * NconstT);
	memcpy(xold_h, x4_h, sizeof(double4) * NconstT);
	memcpy(vold_h, v4_h, sizeof(double4) * NconstT);
	memcpy(rcritv_h, rcrit_h, sizeof(double) * NconstT * P.SLevels);
	memcpy(rcritb_h, rcrit_h, sizeof(double) * NconstT * P.SLevels);
	memcpy(rcritvb_h, rcrit_h, sizeof(double) * NconstT * P.SLevels);
	memcpy(rcritbb_h, rcrit_h, sizeof(double) * NconstT * P.SLevels);
	memcpy(rcritvbb_h, rcrit_h, sizeof(double) * NconstT * P.SLevels);
	memcpy(indexb_h, index_h, sizeof(int) * NconstT);
	memcpy(indexbb_h, index_h, sizeof(int) * NconstT);

#endif


#if def_TTV > 0

  #if MCMC_BLOCK == 5
	memset(elementsG_h, 0, NconstT * sizeof(elements8));
	memset(elementsGh_h, 0, NconstT * sizeof(elements8));
	memset(elementsD_h, 0, NconstT * sizeof(elements8));
	memset(elementsMean_h, 0, NconstT * sizeof(elements8));
	memset(elementsVar_h, 0, NconstT * sizeof(elements8));
  #endif
  #if MCMC_BLOCK == 7
	memset(elementsStep_h, 0, NconstT * sizeof(elementsS));
	memset(elementsHist_h, 0, (Nst + N_h[0] * P.mcmcNE) / (N_h[0] * P.mcmcNE + 1) * N_h[0] * MCMC_NH * sizeof(elementsH));
  #endif
  #if MCMC_NCOV > 0
  #endif
#endif

	int error;

	error = 0;
	fprintf(masterfile,"cudaMemcopy error = %d = %s\n",error, "-");
	if(error != 0){
		printf("cudaMemcopy error = %d = %s\n",error, "-");
		return 0;
	}

	return 1;
}

// ************************************** //
//This function reads the initial conditions from the IC file.
//Authors: Simon Grimm, Joachim Stadel
//March 2014
// *****************************************
 int Data::readic(int st){

	int N = N_h[st];
	int Nsmall = Nsmall_h[st];
	int NBS = NBS_h[st];

	FILE *infile;	

	double AU = def_AU * 100.0; // in cm
	double Solarmass = def_Solarmass * 1000.0; //in g
	if(P.mcmcRestart == 0){
		if(P.FormatP == 1 || P.tRestart == 0){
			if(P.OutBinary > 0 && P.tRestart > 0){
				infile = fopen(GSF[st].inputfilename, "rb");
			}
			else{
				infile = fopen(GSF[st].inputfilename, "r");
			}
		}
		else{
			infile = NULL;
		}
//		printf("Read file %s %d %d\n", GSF[st].inputfilename, N, Nsmall);
	}
	else{
		if(st == 0){
			MCMCRestartFile = fopen("MCMCR.dat", "r");
			printf("Use MCMCR.dat file\n");
			if(MCMCRestartFile == NULL){
				printf("Error, file MCMCR.dat does not exist, needed for mcmc restart options");
				return 0;
			}
		}
		infile = MCMCRestartFile;
	}


	int ii = 0;
	int iismall = 0;
	MaxIndex = 0;
	
	double skip, test;
	double4 x, v;
	double rcrit;
	double4 spin;
	double3 love;
	double3 migration;
	int index;
	float4 aelimits;
	unsigned long long enccountT;
	double mJ = 0.0;	//Jacoby mass
	if(P.tRestart == 0 || def_TTV > 0){
		int er = 0;
		for(int i = 0; i < N + Nsmall; ++i){
			x = x4_h[i + NBS];
			v = v4_h[i + NBS];
			rcrit = rcrit_h[i + NBS];
			spin = spin_h[i + NBS];
			love = love_h[i + NBS];
			if(P.UseMigrationForce > 0){
				migration = migration_h[i + NBS];
			}
			test = test_h[i + NBS];
			enccountT = enccountT_h[i + NBS];
			//index = index_h[i + NBS];
			index = i + st * def_MaxIndex;
			aelimits = aelimits_h[i + NBS];
			int keplerian = 0;
			int convertPToA = 0;
			double p = 0.0;
			int convertTToM = 0;
			double T = 0.0;
			int kepCheck = 0;
			int cartCheck = 0;
#if def_TTV > 0
			double4 elementsA = elementsA_h[i + NBS];
			double4 elementsB = elementsB_h[i + NBS];
			double4 elementsT = elementsT_h[i + NBS];
			double4 elementsSpin = elementsSpin_h[i + NBS];
			elements10  elementsL = elementsL_h[i + NBS];
			double elementsSA = elementsSA_h[st];
			double4 elementsP = elementsP_h[st];
#endif

			for(int f = 0; f < def_Ninformat; ++f){
				if(GSF[st].informat[f] == 1){
					//x
					er = fscanf (infile, "%lf",&x.x);
					cartCheck += 1;
				}
				if (GSF[st].informat[f] == 2){
					//y
					er = fscanf (infile, "%lf",&x.y);
					cartCheck += 2;
				}
				if (GSF[st].informat[f] == 3){
					//z
					er = fscanf (infile, "%lf",&x.z);
					cartCheck += 4;
				}
				if (GSF[st].informat[f] == 4){
					//m
					er = fscanf (infile, "%lf",&x.w);
				}
				if (GSF[st].informat[f] == 5){
					//vx
					er = fscanf (infile, "%lf",&v.x);
					cartCheck += 8;
				}
				if (GSF[st].informat[f] == 6){
					//vy
					er = fscanf (infile, "%lf",&v.y);
					cartCheck += 16;
				}
				if (GSF[st].informat[f] == 7){
					//vz
					er = fscanf (infile, "%lf",&v.z);
					cartCheck += 32;
				}
				if (GSF[st].informat[f] == 8){
					//r
					er = fscanf (infile, "%lf",&v.w);
				}
				if (GSF[st].informat[f] == 9){
					//default rho
					er = fscanf (infile, "%lf",&rho[st]);
				}
				if (GSF[st].informat[f] == 10){
					//Sx
					er = fscanf (infile, "%lf",&spin.x);
#if def_TTV > 0
					elementsSpin.y = spin.x;
#endif
				}
				if (GSF[st].informat[f] == 11){
					//Sy
					er = fscanf (infile, "%lf",&spin.y);
#if def_TTV > 0
					elementsSpin.y = spin.y;
#endif
				}
				if (GSF[st].informat[f] == 12){
					//Sz
					er = fscanf (infile, "%lf",&spin.z);
#if def_TTV > 0
					elementsSpin.y = spin.z;
#endif
				}
				if (GSF[st].informat[f] == 13){
					//index
					er = fscanf (infile, "%d",&index);
				}
				if (GSF[st].informat[f] == 14){
					er = fscanf (infile, "%lf",&skip);
				}
				if (GSF[st].informat[f] == 15) er = fscanf (infile, "%f",&aelimits.x);	//amin
				if (GSF[st].informat[f] == 16) er = fscanf (infile, "%f",&aelimits.y);
				if (GSF[st].informat[f] == 17) er = fscanf (infile, "%f",&aelimits.z);
				if (GSF[st].informat[f] == 18) er = fscanf (infile, "%f",&aelimits.w);
				if (GSF[st].informat[f] == 19){
					if(ict_h[st] == 0){
						er = fscanf (infile, "%lf",&ict_h[st]);
					}
					else{
						er = fscanf (infile, "%lf",&skip);
					}
				}
				if (GSF[st].informat[f] == 20) er = fscanf (infile, "%lf",&love.x);
				if (GSF[st].informat[f] == 21) er = fscanf (infile, "%lf",&love.y);
				if (GSF[st].informat[f] == 22) er = fscanf (infile, "%lf",&love.z);
				if (GSF[st].informat[f] == 23){
					//a
					er = fscanf (infile, "%lf",&x.x);
#if def_TTV > 0
					elementsA.x = x.x;
#endif
					keplerian = 1;
					kepCheck += 1;
				}
				if (GSF[st].informat[f] == 24){
					//e
					er = fscanf (infile, "%lf",&x.y);
#if def_TTV > 0
					elementsA.y = x.y;
#endif
					keplerian = 1;
					kepCheck += 2;
				}
				if (GSF[st].informat[f] == 25){
					//inc
					er = fscanf (infile, "%lf",&x.z);
					if(P.AngleUnits == 1) x.z = x.z / 180.0 * M_PI;
#if def_TTV > 0
					elementsA.z = x.z;
#endif
					keplerian = 1;
					kepCheck += 4;
				}
				if (GSF[st].informat[f] == 26){
					//Omega
					er = fscanf (infile, "%lf",&v.x);
					if(P.AngleUnits == 1) v.x = v.x / 180.0 * M_PI;
#if def_TTV > 0
					elementsB.x = v.x;
#endif
					keplerian = 1;
					kepCheck += 8;
				}
				if (GSF[st].informat[f] == 27){
					//w
					er = fscanf (infile, "%lf",&v.y);
					if(P.AngleUnits == 1) v.y = v.y / 180.0 * M_PI;
#if def_TTV > 0
					elementsB.y = v.y;
#endif
					keplerian = 1;
					kepCheck += 16;
				}
				if (GSF[st].informat[f] == 28){
					//M
					er = fscanf (infile, "%lf",&v.z);
					if(P.AngleUnits == 1) v.z = v.z / 180.0 * M_PI;
#if def_TTV > 0
					elementsB.z = v.z;
#endif
					keplerian = 1;
					kepCheck += 32;
				}
				if (GSF[st].informat[f] == 38){
					//P
					er = fscanf (infile, "%lf",&p);
#if def_TTV > 0
					elementsT.z = p;
					elementsT.w = 0.0;
#endif
					keplerian = 1;
					convertPToA = 1;
					kepCheck += 1;
				}
				if (GSF[st].informat[f] == 40){
					//T
					er = fscanf (infile, "%lf",&T);
#if def_TTV > 0
					elementsT.x = T;
					elementsT.y = 0.0;
#endif
					keplerian = 1;
					convertTToM = 1;
					kepCheck += 32;
				}
				if (GSF[st].informat[f] == 42){
					//Rcrit
					er = fscanf (infile, "%lf",&rcrit);
				}
				if (GSF[st].informat[f] == 44){
					//Ic
					er = fscanf (infile, "%lf",&spin.w);
				}
				if (GSF[st].informat[f] == 45){
					//test
					er = fscanf (infile, "%lf",&test);
				}
				if (GSF[st].informat[f] == 46){
					//encc
					er = fscanf (infile, "%llu",&enccountT);
				}
#if def_TTV > 0
				if (GSF[st].informat[f] == 29){
					er = fscanf (infile, "%lf",&elementsL.a);	//aL
				}
				if (GSF[st].informat[f] == 30){
					er = fscanf (infile, "%lf",&elementsL.e);	//eL
				}
				if (GSF[st].informat[f] == 31){
					er = fscanf (infile, "%lf",&elementsL.inc);	//incL
					if(P.AngleUnits == 1) elementsL.inc = elementsL.inc / 180.0 * M_PI;
				}
				if (GSF[st].informat[f] == 32){
					 er = fscanf (infile, "%lf",&elementsL.m);	//mL
				}
				if (GSF[st].informat[f] == 33){
					er = fscanf (infile, "%lf",&elementsL.O);	//OmegaL
					if(P.AngleUnits == 1) elementsL.O = elementsL.O / 180.0 * M_PI;
				}
				if (GSF[st].informat[f] == 34){
					er = fscanf (infile, "%lf",&elementsL.w);	//wL
					if(P.AngleUnits == 1) elementsL.w = elementsL.w / 180.0 * M_PI;
				}
				if (GSF[st].informat[f] == 35){
					er = fscanf (infile, "%lf",&elementsL.M);	//ML
					if(P.AngleUnits == 1) elementsL.M = elementsL.M / 180.0 * M_PI;
				}
				if (GSF[st].informat[f] == 36){
					er = fscanf (infile, "%lf",&elementsL.r);	//rL
				}
				if (GSF[st].informat[f] == 37){
					er = fscanf (infile, "%lf",&elementsSA);	//SAT
				}
				if (GSF[st].informat[f] == 39){
					er = fscanf (infile, "%lf",&elementsL.P);	//PL
				}
				if (GSF[st].informat[f] == 41){
					er = fscanf (infile, "%lf",&elementsL.T);	//TL
				}
				if (GSF[st].informat[f] == 43){
					er = fscanf (infile, "%lf",&elementsP.w);	//gw
				}

#else
				if (GSF[st].informat[f] == 29) er = fscanf (infile, "%lf",&skip);
				if (GSF[st].informat[f] == 30) er = fscanf (infile, "%lf",&skip);
				if (GSF[st].informat[f] == 31) er = fscanf (infile, "%lf",&skip);
				if (GSF[st].informat[f] == 32) er = fscanf (infile, "%lf",&skip);
				if (GSF[st].informat[f] == 33) er = fscanf (infile, "%lf",&skip);
				if (GSF[st].informat[f] == 34) er = fscanf (infile, "%lf",&skip);
				if (GSF[st].informat[f] == 35) er = fscanf (infile, "%lf",&skip);
				if (GSF[st].informat[f] == 36) er = fscanf (infile, "%lf",&skip);
				if (GSF[st].informat[f] == 37) er = fscanf (infile, "%lf",&skip);
				if (GSF[st].informat[f] == 39) er = fscanf (infile, "%lf",&skip);
				if (GSF[st].informat[f] == 41) er = fscanf (infile, "%lf",&skip);
				if (GSF[st].informat[f] == 43) er = fscanf (infile, "%lf",&skip);
#endif
				if(P.UseMigrationForce > 0){	
					if (GSF[st].informat[f] == 49) er = fscanf (infile, "%lf",&migration.x);
					if (GSF[st].informat[f] == 50) er = fscanf (infile, "%lf",&migration.y);
					if (GSF[st].informat[f] == 51) er = fscanf (infile, "%lf",&migration.z);
				}
				else{
					if (GSF[st].informat[f] == 49) er = fscanf (infile, "%lf",&skip);
					if (GSF[st].informat[f] == 50) er = fscanf (infile, "%lf",&skip);
					if (GSF[st].informat[f] == 51) er = fscanf (infile, "%lf",&skip);

				}
				if (GSF[st].informat[f] == 0){
				}
				if (GSF[st].informat[f] < 0 || GSF[st].informat[f] > 51){
					printf("Error, initial condition file format is not valid, %d\n", GSF[st].informat[f]);
					return 0;
				}
			}

			if(dayUnit == 1) x.w *= def_Kg;
			if(convertPToA == 1){
				mJ += x.w;
				double mu = def_ksq * (Msun_h[st].x + mJ);

				volatile double a3 = p * p * dayUnit * dayUnit * mu / (4.0 * M_PI * M_PI);
				double a = cbrt(a3);
				x.x = a;
#if def_TTV > 0
				elementsA.x = a;
//printf("read a %d %.30g %.30g %.30g %.30g %.30g\n", i, p, mu, a, p * p * dayUnit * dayUnit * mu, a3);
#endif
			}
			else{
				// a to p
				mJ += x.w;
				double mu = def_ksq * (Msun_h[st].x + mJ);
				double p2 = x.x * x.x * x.x * 4.0 * M_PI * M_PI / (dayUnit * dayUnit * mu);
				p = sqrt(p2);
				//double a = x.x;
#if def_TTV > 0
				elementsT.z = p;
				elementsT.w = 0.0;
#endif
//printf("read p %d %.30g %.30g %.30g %.30g\n", i, p, mu, a, p2);

			}

			if(convertTToM == 1){

				double w = v.y;
				double e = x.y;

				double nu = M_PI * 0.5 - w;	//true anomaly at first transit
				double ee2 = e * e;
				double ee4 = ee2 * ee2;
				//double time = time_h[0] - dt_h[0] / dayUnit;
				double time = ict_h[0] * 365.25;
				//compute Mean Anomaly of the first transit
				double Mt = nu - 2.0 * e * sin(nu) + (3.0 * 0.25 * ee2 + 0.125 * ee4) * sin(2.0 * nu) - 1.0 / 3.0 * e * ee2 * sin(3.0 * nu) + 5.0/32.0 * ee4 * sin(4.0 * nu);
				double M = -(T - time) / p * 2.0 * M_PI + Mt;
//printf("T to M %.20g %.20g %.20g %.20g %.20g %.20g\n", time, nu, Mt, T, p, M);
				M = fmod(M, 2.0 * M_PI);
				if(M < 0.0) M += 2.0 * M_PI;

				v.z = M;
#if def_TTV > 0
				elementsB.z = M;
#endif
			}
			else{
#if def_TTV > 0

				double w = v.y;
				double e = x.y;
				double M = v.z;

				double nu = M_PI * 0.5 - w;	//true anomaly at first transit
				double ee2 = e * e;
				double ee4 = ee2 * ee2;
				//double time = time_h[0] - dt_h[0] / dayUnit;
				double time = ict_h[0] * 365.25;
				//compute Mean Anomaly of the first transit
				double Mt = nu - 2.0 * e * sin(nu) + (3.0 * 0.25 * ee2 + 0.125 * ee4) * sin(2.0 * nu) - 1.0 / 3.0 * e * ee2 * sin(3.0 * nu) + 5.0/32.0 * ee4 * sin(4.0 * nu);

//printf("M to T %g %g %g %g %g %g\n", M, time, nu, Mt, T, p);

				double T = -(M - Mt) / (2.0 * M_PI) * p + time;
				elementsT.x = T;
				elementsT.y = 0.0;
#endif
			}
			if(keplerian == 0){
				if(cartCheck != 63 || kepCheck != 0){
					printf("Error, initial conditions are not complete. Must include x, y, z, vx, vy, vz \n");
					return 0;
				}
			}
			if(keplerian == 1){
				if(kepCheck != 63 || cartCheck != 0){
					printf("Error, initial conditions are not complete. Must include a (or P), e, inc, O, w, M (or T)\n");
					return 0;
				}

#if def_TTV > 0
				elementsA.w = x.w;		//m
				elementsB.w = v.w;		//r
//printf("read elements %d %.20g %.20g %.20g\n",ii, elementsA.w, elementsA.x, elementsA.y); 
#endif	
				KepToCart(x, v, Msun_h[st].x);
			}
			if(index < 0) index *= -1;
			if(v.w == 0){
				v.w = cbrt((x.w * 0.75 ) / (M_PI * rho[st] * AU * AU * AU / Solarmass));
			}

			//avoid max for long long int
			if(index > MaxIndex){
				MaxIndex = index;
			}
			
			int NBSN = NBS;
			if(x.w >= 0.0 && x.w <= P.MinMass && P.UseTestParticles > 0){
				NBSN += N - ii + iismall; //shift test particles to the end of the arrays
			}
			else{
				NBSN -= iismall;
			}

			x4_h[ii + NBSN] = x;
			v4_h[ii + NBSN] = v;
			rcrit_h[ii + NBSN] = rcrit;
			spin_h[ii + NBSN] = spin;
			love_h[ii + NBSN] = love;
			if(P.UseMigrationForce > 0){
				migration_h[ii + NBSN] = migration;
			}
			if(Nst == 1) index_h[ii + NBSN] = index;
			else index_h[ii + NBSN] = index % def_MaxIndex + def_MaxIndex * st;
			aelimits_h[ii + NBSN] = aelimits;
			enccountT_h[ii + NBSN] = enccountT;
			test_h[ii + NBSN] = test;
#if def_TTV > 0
			elementsA_h[ii + NBSN] = elementsA;
			elementsB_h[ii + NBSN] = elementsB;
			elementsT_h[ii + NBSN] = elementsT;
			elementsSpin_h[ii + NBSN] = elementsSpin;
			elementsL_h[ii + NBSN] = elementsL;
			int iT = st / (Nst / MCMC_NT);			//index of temperature in parallel tempering
			 
			elementsSA_h[st] = elementsSA * pow(sqrt(2.0), iT);
			elementsP_h[st] = elementsP;
#endif
			++ii;
			if(x.w >= 0 && x.w <= P.MinMass && P.UseTestParticles > 0) ++iismall;
		}// end of particle loop
		//check now if the file is finished
		if(def_TTV == 0){
			er = fscanf (infile, "%lf",&skip);
			if(er != -1 && st == Nst -1){
				printf("Error, initial condition file format is not correct\n");
				return 0;
			}
		}

	}
	else{
#if def_TTV > 0
		printf("Restart for TTV not possible\n");
		return 0;

#endif
		//read from restart time step
		double Et;
		if(P.OutBinary == 0){
			char Ets[160]; //exact time at restart time step, must be the same format as the coordinate output
			sprintf(Ets, "%.16g", (P.tRestart * idt_h[st] + ict_h[st] * 365.25) / 365.25);
			Et = atof(Ets);
		}
		else{
			Et = (P.tRestart * idt_h[st] + ict_h[st] * 365.25) / 365.25;
		}

		double time = 0.0;
		double aecountf = 0.0;
		double aecountTf = 0.0;
		unsigned int aecount;
		unsigned long long aecountT;
		unsigned long long enccountT;

		spin.x = 0.0;
		spin.y = 0.0;
		spin.z = 0.0;
		spin.w = 0.4;
		love.x = 0.0;
		love.y = 0.0;
		love.z = 0.0;
		migration.x = 0.0;
		migration.y = 0.0;
		migration.z = 0.0;
		rcrit = 0.0;

		if(P.FormatP == 1){

			//skip previous time steps
			if(P.FormatT == 0){
				readOutLine(time, index, x, v, spin, love, migration, aelimits, aecountf, aecountTf, enccountT, rcrit, test, infile, st);
//printf("T0 %d %d %g %g | %d %g %g\n", st, 0, time, Et, index, x.w, x.x);
			}
			if(P.FormatT == 1){
				readOutLine(time, index, x, v, spin, love, migration, aelimits, aecountf, aecountTf, enccountT, rcrit, test, infile, st);
				while((time < Et && idt_h[st] > 0) || (time > Et && idt_h[st] < 0)){
					if(time == Et) break;
					int er = readOutLine(time, index, x, v, spin, love, migration, aelimits, aecountf, aecountTf, enccountT, rcrit, test, infile, st);
//printf("T1 %d %d %g %g | %d %g %g\n", st, 0, time, Et, index, x.w, x.x);
					if(er <= 0){
						break;
					}
				}
			}


			//skip previous simulation data
			if(P.FormatS == 1){
				for(int i = 0; i < NBS; ++i){
					readOutLine(time, index, x, v, spin, love, migration, aelimits, aecountf, aecountTf, enccountT, rcrit, test, infile, st);
//printf("S %d %d %g %g | %d %g %g\n", st, i, time, Et, index, x.w, x.x);
				}
			}

			int iismall = 0;
			for(int i = 0; i < N + Nsmall; ++i){
				if(i > 0) readOutLine(time, index, x, v, spin, love, migration, aelimits, aecountf, aecountTf, enccountT, rcrit, test, infile, st);
//printf("r %d %d %g %g | %d %g %g\n", st, i, time, Et, index, x.w, x.x);

				if(P.FormatS == 0) index += def_MaxIndex * st;
				aecount = (unsigned int)(aecountf * P.ci);
				unsigned long long tt = P.tRestart - P.tRestart % P.ci;
				aecountT = (unsigned long long)(aecountTf * tt);

				//avoid max for long long int
				if(index > MaxIndex){
					MaxIndex = index;
				}

				int NBSN = NBS;
				if(x.w >= 0.0 && x.w <= P.MinMass && P.UseTestParticles > 0){
					NBSN += N - i + iismall; //shift test particles to the end of the arrays
				}
				else{
					NBSN -= iismall;
				}
				index_h[ii + NBSN] = index;
				x4_h[ii + NBSN] = x;
				v4_h[ii + NBSN] = v;
				rcrit_h[ii + NBSN] = rcrit;
				spin_h[ii + NBSN] = spin;
				love_h[ii + NBSN] = love;
				if(P.UseMigrationForce > 0){
					migration_h[ii + NBSN] = migration;
				}
				aelimits_h[ii + NBSN] = aelimits;
				enccountT_h[ii + NBSN] = enccountT;
				aecount_h[ii + NBSN] = aecount;
				aecountT_h[ii + NBSN] = aecountT;
				test_h[ii + NBSN] = test;
				++ii;
				if(x.w >= 0 && x.w <= P.MinMass && P.UseTestParticles > 0) ++iismall;
			}
		}
		if(P.FormatP == 0){
			ii = 0;
			FILE *OrigInfile;	
			char Origfilename[512];
			sprintf(Origfilename, "%s%s", GSF[st].path, GSF[st].Originputfilename);
			OrigInfile = fopen(Origfilename, "r");

			FILE *fragmentsfile;
			if(P.UseSmallCollisions > 0 || P.CreateParticles > 0){
				fragmentsfile = fopen(GSF[st].fragmentfilename, "r");
			}
			else{
				fragmentsfile = NULL;
			}

			int iismall = 0;
			int index;

			for(int k = 0; k < 1000000000; ++k){
				int i = k;
				double skip = 0.0;
				int eri = 1;
				for(int f = 0; f < def_Ninformat; ++f){
					if(GSF[st].informat[f] == 13){
						eri = fscanf (OrigInfile, "%d",&i);
					}
					else if(GSF[st].informat[f] > 0){
						eri = fscanf (OrigInfile, "%lf",&skip);
					}
				}
				if(eri < 0){
					if(P.UseSmallCollisions > 0 || P.CreateParticles > 0){
//printf("Search for fragments %s\n", GSF[st].fragmentfilename);
						double ttime, mm;
						double skip;
						eri = fscanf(fragmentsfile, "%lf", &ttime);
						eri = fscanf(fragmentsfile, "%d", &i);
						eri = fscanf(fragmentsfile, "%lf", &mm);

						for(int jj = 0; jj < 11; ++jj){
							eri = fscanf(fragmentsfile, "%lf", &skip);
						}
						if(eri <= 0){
							break;
						}

						if(ttime > Et) continue;
					}
					else{
						break;
					}
				}
		
				int er = 0;
				char infilename[384];
				if(P.OutBinary == 0){
					sprintf(infilename, "%sOut%s_p%.6d.dat", GSF[st].path, GSF[st].X, i);
					infile = fopen(infilename, "r");
				}
				else{
					sprintf(infilename, "%sOut%s_p%.6d.bin", GSF[st].path, GSF[st].X, i);
					infile = fopen(infilename, "rb");
				}
//printf("Read file %s %d %d %d\n", infilename, ii, N, Nsmall);
				if(infile == NULL) continue;
	
				//skip previous time steps
				er = readOutLine(time, index, x, v, spin, love, migration, aelimits, aecountf, aecountTf, enccountT, rcrit, test, infile, st);
//printf("T0 %d %d %g %g | %d %g %g\n", st, 0, time, Et, index, x.w, x.x);
				while((time < Et && idt_h[st] > 0) || (time > Et && idt_h[st] < 0)){
					if(time == Et){
						break;
					}
					er = readOutLine(time, index, x, v, spin, love, migration, aelimits, aecountf, aecountTf, enccountT, rcrit, test, infile, st);
//printf("T1 %d %d %g %g | %d %g %g\n", st, 0, time, Et, index, x.w, x.x);
					if(er <= 0){
						break;
					}
				}
				if(er <= 0){
					continue;
				}

				if(P.FormatS == 0) index += def_MaxIndex * st;
				aecount = (unsigned int)(aecountf * P.ci);
				unsigned long long tt = P.tRestart - P.tRestart % P.ci;
				aecountT = (unsigned long long)(aecountTf * tt);

				//avoid max for long long int
				if(index > MaxIndex){
					MaxIndex = index;
				}

				int NBSN = NBS;
				if(x.w >= 0.0 && x.w <= P.MinMass && P.UseTestParticles > 0){
					NBSN += N - ii + iismall; //shift test particles to the end of the arrays
				}
				else{
					NBSN -= iismall;
				}
				index_h[ii + NBSN] = index;
				x4_h[ii + NBSN] = x;
				v4_h[ii + NBSN] = v;
				rcrit_h[ii + NBSN] = rcrit;
				spin_h[ii + NBSN] = spin;
				love_h[ii + NBSN] = love;
				if(P.UseMigrationForce > 0){
					migration_h[ii + NBSN] = migration;
				}
				aelimits_h[ii + NBSN] = aelimits;
				enccountT_h[ii + NBSN] = enccountT;
				aecount_h[ii + NBSN] = aecount;
				aecountT_h[ii + NBSN] = aecountT;
				test_h[ii + NBSN] = test;

				++ii;
				if(x.w >= 0 && x.w <= P.MinMass && P.UseTestParticles > 0) ++iismall;

				fclose(infile);
//printf("%d %d %d\n", ii, iismall, N + Nsmall);
				if(ii == N + Nsmall) break;
			}
			fclose(OrigInfile);
			if(P.UseSmallCollisions > 0 || P.CreateParticles > 0){
				fclose(fragmentsfile);
			}

		}
	}
	if(P.mcmcRestart == 0){
		if(P.FormatP == 1 || P.tRestart == 0) fclose(infile);
	}
	else{
		if(st == Nst - 1){
			fclose(infile);
		}
	}
	return ii;
} 


// *************************************
//This function converts Keplerian Elements into Cartesian Coordinates
 void Data::KepToCart(double4 &x, double4 &v, double Msun){

	double a = x.x;
	double e = x.y;
	double inc = x.z;
	double Omega = v.x;
	double w = v.y;
	double M = v.z;
//printf("A KtoC m:%g r:%g a:%g e:%g i:%g O:%g w:%g M:%g\n", x.w, v.w, x.x, x.y, x.z, v.x ,v.y, v.z);

	double mu = def_ksq * (Msun + x.w);

	double E;
	if(e < 1.0 - 1.0e-10){	
		//Eccentric Anomaly
		E = M + e * 0.5;
		double Eold = E;
		for(int j = 0; j < 32; ++j){
			E = E - (E - e * sin(E) - M) / (1.0 - e * cos(E));
			if(fabs(E - Eold) < 1.0e-15) break;
			Eold = E;
		}
	}
	else if(e > 1.0 + 1.0e-10){
		//hyperbolic
		//E is assumed to be the hyperbolic eccentricity 
		E = M;
		double Eold = E;
		for(int j = 0; j < 32; ++j){
			E = E + (E - e * sinh(E) + M) / (e * cosh(E) - 1.0);
			if(fabs(E - Eold) < 1.0e-15) break;
			Eold = E;
		}

	}
	else{
		//parabolic, solve Barkers equation 
		// M = D + D^3 / 3, 
		// use cot(s) = 1.5 * M  -> s = pi / 2 - atan(1.5 * M)

		//double s = M_PI * 0.5 - atan(1.5 * M);
		E = M;
		double Eold = E;
		for(int j = 0; j < 32; ++j){
			E = E - (E + E * E * E / 3.0 - M) / (1.0 + E * E);
			if(fabs(E - Eold) < 1.0e-15) break;
			Eold = E;
		}

	}


	double cw = cos(w);
	double sw = sin(w);
	double cOmega = cos(Omega);
	double sOmega = sin(Omega);
	double ci = cos(inc);
	double si = sin(inc);

	double Px = cw * cOmega - sw * ci * sOmega;
	double Py = cw * sOmega + sw * ci * cOmega;
	double Pz = sw * si;

	double Qx = -sw * cOmega - cw * ci * sOmega;
	double Qy = -sw * sOmega + cw * ci * cOmega;
	double Qz = cw * si;

	double cE = cos(E);
	double sE = sin(E);

	double t0, t1, t2;

	if(e < 1.0 - 1.0e-10){
		//elliptic

		//double r = a * ( 1.0 - e * cE);
		//double r = a * (1.0 - e*e)/(1.0 + e *cos(Theta));
		//double t1 = r * cos(Theta); 
		//double t2 = r * sin(Theta); 
		t1 = a * (cE - e);
		t2 = a * sqrt(1.0 - e * e) * sE;
	}
	else if(e > 1.0 + 1.0e-10){
		//hyperbolic
		//double r = a * (1.0 - e*e)/(1.0 + e *cos(Theta));
		//or
		//double r = a * ( 1.0 - e * cosh(E));
		//t1 = r * cos(Theta); 
		//t2 = r * sin(Theta); 
		t1 = a * (cosh(E) - e);
		t2 = -a * sqrt(e * e - 1.0) * sinh(E);
	}
	else{
		//parabolic
		// a is assumed to be q, p = 2q, p = h^2/mu
		double Theta = 2.0 * atan(E);
		double r = 2 * a /(1.0 + cos(Theta));
		t1 = r * cos(Theta);
		t2 = r * sin(Theta);
	}


	x.x = t1 * Px + t2 * Qx;
	x.y = t1 * Py + t2 * Qy;
	x.z = t1 * Pz + t2 * Qz;

	if(e < 1.0 - 1.0e-10){
		//elliptic
		t0 = 1.0 / (1.0 - e * cE) * sqrt(mu / a);
		t1 = -sE;
		t2 = sqrt(1.0 - e * e) * cE;
	}
	else if(e > 1.0 + 1.0e-10){
		//hyperbolic
		//double r = a * (1.0 - e*e)/(1.0 + e *cos(Theta));
		double r = a * ( 1.0 - e * cosh(E));
		t0 = sqrt(-mu * a) / r;
		t1 = -sinh(E);
		t2 = sqrt(e * e - 1.0) * cosh(E);
	}
	else{
		//parabolic
		double Theta = 2.0 * atan(E);
		t0 = mu / sqrt(2.0 * a * mu);
		t1 = -sin(Theta);
		t2 = 1.0 + cos(Theta);
	}



	v.x = t0 * (t1 * Px + t2 * Qx);
	v.y = t0 * (t1 * Py + t2 * Qy);
	v.z = t0 * (t1 * Pz + t2 * Qz);
//printf("B KtoC m:%g r:%g x:%g y:%g z:%g vx:%g vy:%g vz:%g\n", x.w, v.w, x.x, x.y, x.z, v.x ,v.y, v.z);
}

// **************************************
//This function converts heliocentric coordinates to democratic coordinates.
 void Data::HelioToDemo(double4 *x4_h, double4 *v4_h, double Msun, int N){

	double mtot = 0.0;
	double3 vcom;
	vcom.x = 0.0;
	vcom.y = 0.0;
	vcom.z = 0.0;
	
	for(int i = 0; i < N; ++i){
		if(x4_h[i].w > 0.0){
			double m = x4_h[i].w;
			mtot += m;
			vcom.x += m * v4_h[i].x;
			vcom.y += m * v4_h[i].y;
			vcom.z += m * v4_h[i].z;
		}
	}
	mtot += Msun;
	vcom.x /= mtot;
	vcom.y /= mtot;
	vcom.z /= mtot;

	for(int i = 0; i < N; ++i){
		v4_h[i].x -= vcom.x;
		v4_h[i].y -= vcom.y;
		v4_h[i].z -= vcom.z;
	}
}
// This function converts heliocentric coordinates to barycentric coordinates.
// The zeroth body must be the central star here
 void Data::HelioToBary(double4 *x4_h, double4 *v4_h, double Msun, int N){

	double mtot = 0.0;
	double3 vcom;
	double3 xcom;
	xcom.x = 0.0;
	xcom.y = 0.0;
	xcom.z = 0.0;
	vcom.x = 0.0;
	vcom.y = 0.0;
	vcom.z = 0.0;
	
	for(int i = 0; i < N; ++i){
//printf("A HtB %g %g %g %g %g %g %g %g\n", x4_h[i].w, v4_h[i].w, x4_h[i].x, x4_h[i].y, x4_h[i].z, v4_h[i].x ,v4_h[i].y, v4_h[i].z);
		if(x4_h[i].w > 0.0){
			double m = x4_h[i].w;
			mtot += m;
			xcom.x += m * x4_h[i].x;
			xcom.y += m * x4_h[i].y;
			xcom.z += m * x4_h[i].z;
			vcom.x += m * v4_h[i].x;
			vcom.y += m * v4_h[i].y;
			vcom.z += m * v4_h[i].z;
		}
	}
	xcom.x /= mtot;
	xcom.y /= mtot;
	xcom.z /= mtot;
	vcom.x /= mtot;
	vcom.y /= mtot;
	vcom.z /= mtot;

	for(int i = 0; i < N; ++i){
		x4_h[i].x -= xcom.x;
		x4_h[i].y -= xcom.y;
		x4_h[i].z -= xcom.z;
		v4_h[i].x -= vcom.x;
		v4_h[i].y -= vcom.y;
		v4_h[i].z -= vcom.z;
//printf("B HtB %g %g %g %g %g %g %g %g\n", x4_h[i].w, v4_h[i].w, x4_h[i].x, x4_h[i].y, x4_h[i].z, v4_h[i].x ,v4_h[i].y, v4_h[i].z);
	}
}
// **************************************
//This function converts democratic coordinates to heliocentric coordinates.
 void Data::DemoToHelio(double4 *x4_h, double4 *v4_h, double4 *v4Helio_h, double Msun, int N){

	double3 vcom;
	vcom.x = 0.0;
	vcom.y = 0.0;
	vcom.z = 0.0;

	for(int i = 0; i < N; ++i){
		if(x4_h[i].w > 0.0){
			vcom.x += x4_h[i].w * v4_h[i].x;
			vcom.y += x4_h[i].w * v4_h[i].y;
			vcom.z += x4_h[i].w * v4_h[i].z;
		}
	}
	vcom.x /= Msun;
	vcom.y /= Msun;
	vcom.z /= Msun;

	for(int i = 0; i < N; ++i){
		v4Helio_h[i].x = v4_h[i].x + vcom.x;
		v4Helio_h[i].y = v4_h[i].y + vcom.y;
		v4Helio_h[i].z = v4_h[i].z + vcom.z;
		v4Helio_h[i].w = v4_h[i].w;
	}

}
// **************************************
//This function converts barycentric coordinates to heliocentric coordinates.
// The zeroth body must be the cetnral star here
 void Data::BaryToHelio(double4 *x4_h, double4 *v4_h, double Msun, int N){

	double3 xcom;
	double3 vcom;
	xcom.x = 0.0;
	xcom.y = 0.0;
	xcom.z = 0.0;
	vcom.x = 0.0;
	vcom.y = 0.0;
	vcom.z = 0.0;

	for(int i = 0; i < N; ++i){
		if(x4_h[i].w > 0.0){
			xcom.x += x4_h[i].w * x4_h[i].x;
			xcom.y += x4_h[i].w * x4_h[i].y;
			xcom.z += x4_h[i].w * x4_h[i].z;
			vcom.x += x4_h[i].w * v4_h[i].x;
			vcom.y += x4_h[i].w * v4_h[i].y;
			vcom.z += x4_h[i].w * v4_h[i].z;
		}
	}
	xcom.x /= x4_h[0].w;
	xcom.y /= x4_h[0].w;
	xcom.z /= x4_h[0].w;
	vcom.x /= x4_h[0].w;
	vcom.y /= x4_h[0].w;
	vcom.z /= x4_h[0].w;

	for(int i = 0; i < N; ++i){
		x4_h[i].x += xcom.x;
		x4_h[i].y += xcom.y;
		x4_h[i].z += xcom.z;
		v4_h[i].x += vcom.x;
		v4_h[i].y += vcom.y;
		v4_h[i].z += vcom.z;
	}
}


// **************************************
//This kernel removes ghost-masses and decreases the number of bodies.
//It also removes bodies wich a semi major axis bigger than Rcut.
//It runs with only one thread ond the GPU, to avoid unnecesary data copies
//Authors: Simon Grimm, Joachim Stadel
//March 2014
// ***************************************
void remove_cpu(double4 *x4_h, double4 *v4_h, double3 *a_h, int *N_h, int *Nsmall_h, int *index_h, double4 *spin_h, double3 *love_h, double3 *migration_h, int *createFlag_h, double *test_h, double *EnergySum_h, double *rcrit_h, double *rcritv_h, int NBS, int st, float4 *aelimits_h, unsigned int *aecount_h, unsigned int *enccount_h, unsigned long long *aecountT_h, unsigned long long *enccountT_h, double *K_h, double *Kold_h, int NB, const int NconstT, const int SLevels, const int UseMigrationForce, const int CreateParticles, double *nafx_h, double *nafy_h, int nafn){
	int NOld;
	int NsmallOld;
	int N = N_h[st];
	int Nsmall = Nsmall_h[st];
	int f = 1;
	int fc = 0;

	while(f == 1 && fc < 100){
		NOld = N;
		NsmallOld = Nsmall;
		f = 0;
		++fc;
		for(int j = 0; j < N; ++j){
			//remove ghost bodies and rearrange arrays
			if(x4_h[j + NBS].w < 0){
				int Na = j + NBS;
				int Nb = N-1 + NBS;
				
				x4_h[Na] = x4_h[Nb];
				v4_h[Na] = v4_h[Nb];

				x4_h[Nb].x = 0.0;
				x4_h[Nb].y = 1.0;
				x4_h[Nb].z = 0.0;
				x4_h[Nb].w = -1.0e-12;
	
				v4_h[Nb].x = 0.0;
				v4_h[Nb].y = 0.0;
				v4_h[Nb].z = 0.0;
				v4_h[Nb].w = 0.0;

				a_h[Na] = a_h[Nb];
				a_h[Nb].x = 0.0;
				a_h[Nb].y = 0.0;
				a_h[Nb].z = 0.0;

				index_h[Na] = index_h[Nb];
				index_h[Nb] = -1;

				spin_h[Na] = spin_h[Nb];
				spin_h[Nb].x = 0.0;
				spin_h[Nb].y = 0.0;
				spin_h[Nb].z = 0.0;
				spin_h[Nb].w = 0.0;
	
				love_h[Na] = love_h[Nb];
				love_h[Nb].x = 0.0;
				love_h[Nb].y = 0.0;
				love_h[Nb].z = 0.0;

				if(UseMigrationForce > 0){
					migration_h[Na] = migration_h[Nb];
					migration_h[Nb].x = 0.0;
					migration_h[Nb].y = 0.0;
					migration_h[Nb].z = 0.0;
				}

				if(CreateParticles > 0){
					createFlag_h[Na] = createFlag_h[Nb];
					createFlag_h[Nb] = -1;
				}

				for(int l = 0; l < SLevels; ++l){
					rcrit_h[Na + l * NconstT] = rcrit_h[Nb + l * NconstT];
					rcritv_h[Na + l * NconstT] = rcritv_h[Nb + l * NconstT];
					rcrit_h[Nb + l * NconstT] = 0.0;
					rcritv_h[Nb + l * NconstT] = 0.0;
				}

				aelimits_h[Na] = aelimits_h[Nb];
				aelimits_h[Nb].x = 0.0f;
				aelimits_h[Nb].y = 0.0f;
				aelimits_h[Nb].z = 0.0f;	
				aelimits_h[Nb].w = 0.0f;

				aecount_h[Na] = aecount_h[Nb];
				aecount_h[Nb] = 0u;
				enccount_h[Na] = enccount_h[Nb];
				enccount_h[Nb] = 0u;
				aecountT_h[Na] = aecountT_h[Nb];
				aecountT_h[Nb] = 0ull;
				enccountT_h[Na] = enccountT_h[Nb];
				enccountT_h[Nb] = 0ull;

				test_h[Na] = test_h[Nb];
				test_h[Nb] = -1.0;

				EnergySum_h[Na] += EnergySum_h[Nb];
				EnergySum_h[Nb] = 0.0;

				for(int i = 0; i < nafn; ++i){
					nafx_h[(Na) * nafn + i] = nafx_h[(Nb) * nafn + i];
					nafy_h[(Na) * nafn + i] = nafy_h[(Nb) * nafn + i];
					nafx_h[(Nb) * nafn + i] = 0.0;
					nafy_h[(Nb) * nafn + i] = 0.0;
				}
#if def_G3 > 0
				for(int i = 0; i < N; ++i){
					K_h[(Na) * NB + i] = K_h[(Nb) * NB + i];
					K_h[i * NB + Na] = K_h[i * NB + (Nb)];
					K_h[(Nb) * NB + i] = 1.0;
					K_h[i * NB + (Nb)] = 1.0;
					Kold_h[(Na) * NB + i] = Kold_h[(Nb) * NB + i];
					Kold_h[i * NB + Na] = Kold_h[i * NB + (Nb)];
					Kold_h[(Nb) * NB + i] = 1.0;
					Kold_h[i * NB + (Nb)] = 1.0;
				}
#endif
				//move Test Particles
				if(Nsmall > 0){
					int Na = N-1 + NBS;
					int Nb = N-1 + NBS + Nsmall;
					
					x4_h[Na] = x4_h[Nb];
					v4_h[Na] = v4_h[Nb];

					x4_h[Nb].x = 0.0;
					x4_h[Nb].y = 1.0;
					x4_h[Nb].z = 0.0;
					x4_h[Nb].w = -1.0e-12;
		
					v4_h[Nb].x = 0.0;
					v4_h[Nb].y = 0.0;
					v4_h[Nb].z = 0.0;
					v4_h[Nb].w = 0.0;

					a_h[Na] = a_h[Nb];
					a_h[Nb].x = 0.0;
					a_h[Nb].y = 0.0;
					a_h[Nb].z = 0.0;

					index_h[Na] = index_h[Nb];
					index_h[Nb] = -1;

					spin_h[Na] = spin_h[Nb];
					spin_h[Nb].x = 0.0;
					spin_h[Nb].y = 0.0;
					spin_h[Nb].z = 0.0;
					spin_h[Nb].w = 0.0;

					love_h[Na] = love_h[Nb];
					love_h[Nb].x = 0.0;
					love_h[Nb].y = 0.0;
					love_h[Nb].z = 0.0;
	
					if(UseMigrationForce > 0){
						migration_h[Na] = migration_h[Nb];
						migration_h[Nb].x = 0.0;
						migration_h[Nb].y = 0.0;
						migration_h[Nb].z = 0.0;
					}

					if(CreateParticles > 0){
						createFlag_h[Na] = createFlag_h[Nb];
						createFlag_h[Nb] = 0;
					}
			
					for(int l = 0; l < SLevels; ++l){
						rcrit_h[Na + l * NconstT] = rcrit_h[Nb + l * NconstT];
						rcritv_h[Na + l * NconstT] = rcritv_h[Nb + l * NconstT];
						rcrit_h[Nb + l * NconstT] = 0.0;
						rcritv_h[Nb + l * NconstT] = 0.0;
					}

					aelimits_h[Na] = aelimits_h[Nb];
					aelimits_h[Nb].x = 0.0f;
					aelimits_h[Nb].y = 0.0f;
					aelimits_h[Nb].z = 0.0f;	
					aelimits_h[Nb].w = 0.0f;

					aecount_h[Na] = aecount_h[Nb];
					aecount_h[Nb] = 0u;
					enccount_h[Na] = enccount_h[Nb];
					enccount_h[Nb] = 0u;
					aecountT_h[Na] = aecountT_h[Nb];
					aecountT_h[Nb] = 0ull;
					enccountT_h[Na] = enccountT_h[Nb];
					enccountT_h[Nb] = 0ull;

					test_h[Na] = test_h[Nb];
					test_h[Nb] = -1.0;

					EnergySum_h[Na] += EnergySum_h[Nb];
					EnergySum_h[Nb] = 0.0;

					for(int i = 0; i < nafn; ++i){
						nafx_h[(Na) * nafn + i] = nafx_h[(Nb) * nafn + i];
						nafy_h[(Na) * nafn + i] = nafy_h[(Nb) * nafn + i];
						nafx_h[(Nb) * nafn + i] = 0.0;
						nafy_h[(Nb) * nafn + i] = 0.0;
					}
				}

				N -= 1;
			}
		}
		for(int j = N; j < N + Nsmall; ++j){
			//remove ghost test particles and rearrange arrays
			if(x4_h[j + NBS].w < 0){

				int Na = j + NBS;
				int Nb = N-1 + NBS + Nsmall;
				x4_h[Na] = x4_h[Nb];
				v4_h[Na] = v4_h[Nb];

				x4_h[Nb].x = 0.0;
				x4_h[Nb].y = 1.0;
				x4_h[Nb].z = 0.0;
				x4_h[Nb].w = -1.0e-12;
	
				v4_h[Nb].x = 0.0;
				v4_h[Nb].y = 0.0;
				v4_h[Nb].z = 0.0;
				v4_h[Nb].w = 0.0;

				a_h[Na] = a_h[Nb];
				a_h[Nb].x = 0.0;
				a_h[Nb].y = 0.0;
				a_h[Nb].z = 0.0;

				index_h[Na] = index_h[Nb];
				index_h[Nb] = -1;

				spin_h[Na] = spin_h[Nb];
				spin_h[Nb].x = 0.0;
				spin_h[Nb].y = 0.0;
				spin_h[Nb].z = 0.0;
				spin_h[Nb].w = 0.0;

				love_h[Na] = love_h[Nb];
				love_h[Nb].x = 0.0;
				love_h[Nb].y = 0.0;
				love_h[Nb].z = 0.0;

				if(UseMigrationForce > 0){
					migration_h[Na] = migration_h[Nb];
					migration_h[Nb].x = 0.0;
					migration_h[Nb].y = 0.0;
					migration_h[Nb].z = 0.0;
				}

				if(CreateParticles > 0){
					createFlag_h[Na] = createFlag_h[Nb];
					createFlag_h[Nb] = -1;
				}

				for(int l = 0; l < SLevels; ++l){
					rcrit_h[Na + l * NconstT] = rcrit_h[Nb + l * NconstT];
					rcritv_h[Na + l * NconstT] = rcritv_h[Nb + l * NconstT];
					rcrit_h[Nb + l * NconstT] = 0.0;
					rcritv_h[Nb + l * NconstT] = 0.0;
				}

				aelimits_h[Na] = aelimits_h[Nb];
				aelimits_h[Nb].x = 0.0f;
				aelimits_h[Nb].y = 0.0f;
				aelimits_h[Nb].z = 0.0f;	
				aelimits_h[Nb].w = 0.0f;

				aecount_h[Na] = aecount_h[Nb];
				aecount_h[Nb] = 0u;
				enccount_h[Na] = enccount_h[Nb];
				enccount_h[Nb] = 0u;
				aecountT_h[Na] = aecountT_h[Nb];
				aecountT_h[Nb] = 0ull;
				enccountT_h[Na] = enccountT_h[Nb];
				enccountT_h[Nb] = 0ull;

				test_h[Na] = test_h[Nb];
				test_h[Nb] = -1.0;

				EnergySum_h[Na] += EnergySum_h[Nb];
				EnergySum_h[Nb] = 0.0;

				for(int i = 0; i < nafn; ++i){
					nafx_h[(Na) * nafn + i] = nafx_h[(Nb) * nafn + i];
					nafy_h[(Na) * nafn + i] = nafy_h[(Nb) * nafn + i];
					nafx_h[(Nb) * nafn + i] = 0.0;
					nafy_h[(Nb) * nafn + i] = 0.0;
				}
				Nsmall -= 1;
			}
		}
		if(NOld != N) f = 1;
		if(NsmallOld != Nsmall) f = 1;
	}
	N_h[st] = N;
	Nsmall_h[st] = Nsmall;
}


// **************************************
//This function prints out data of ejected bodies
//It sets the masses of ejected bodies to zero, this are then later removed
//It Updates the lost Energy term U
//
//Authors: Simon Grimm, Joachim Stadel
//Mai 2015
//****************************************
 void Data::Ejection(){

	FILE *ejectfile;
	FILE *logfile;

	if(Nst == 1) EjectionFlag_m[1] = 1;

	for(int st = 0; st < Nst; ++st){
		if(EjectionFlag_m[st + 1] > 0){
			EjectionFlag_m[st + 1] = 0;

			int NBS = NBS_h[st];

			ejectfile = fopen(GSF[st].ejectfilename, "a");
			logfile = fopen(GSF[st].logfilename, "a");

			if(P.UseMigrationForce > 0){
			}

			memset(Nencpairs_h, 0, sizeof(int));

			int c = 0;
			for(int i = 0; i < N_h[st] + Nsmall_h[st]; ++i){
				c = 0;
				double rsq = x4_h[i + NBS].x*x4_h[i + NBS].x + x4_h[i + NBS].y*x4_h[i + NBS].y + x4_h[i + NBS].z*x4_h[i + NBS].z;
				if(rsq > Rcut_h[st] * Rcut_h[st] && x4_h[i + NBS].w >= 0){
					c = -3;
					if(Nst == 1){
						if(x4_h[i + NBS].w > 0.0){
							printf("Body %d ejected\n", index_h[i + NBS]);
							fprintf(logfile, "Body %d ejected\n", index_h[i + NBS]);
						}
						else{
							printf("Test Particle %d ejected\n", index_h[i + NBS]);
							fprintf(logfile, "Test Particle %d ejected\n", index_h[i + NBS]);
						}
					}
					else{
						if(x4_h[i + NBS].w > 0.0){
							printf("In Simulation %s: Body %d ejected \n", GSF[st].path, index_h[i + NBS] % def_MaxIndex);
							fprintf(logfile, "Body %d ejected\n", index_h[i + NBS] % def_MaxIndex);
						}
						else{
							printf("In Simulation %s: Test Particle %d ejected \n", GSF[st].path, index_h[i + NBS] % def_MaxIndex);
							fprintf(logfile, "Test Particle %d ejected\n", index_h[i + NBS] % def_MaxIndex);
						}
					}
				}
//if(i == 619) printf("ejection %d %g %g %g\n", i, rsq, RcutSun_h[st] * RcutSun_h[st], x4_h[i + NBS].w);
				if( rsq < RcutSun_h[st] * RcutSun_h[st] && x4_h[i + NBS].w >= 0){
					c = -2;
					if(Nst == 1){
						if(x4_h[i + NBS].w > 0.0){
							printf("Body %d too close to central mass -> removed\n", index_h[i + NBS]);
							fprintf(logfile, "Body %d too close to central mass -> removed\n", index_h[i + NBS]);
						}
						else{
							printf("Test Particle %d too close to central mass -> removed\n", index_h[i + NBS]);
							fprintf(logfile, "Test Particle %d too close to central mass -> removed\n", index_h[i + NBS]);
						}
					}
					else{
						if(x4_h[i + NBS].w > 0.0){
							printf("In Simulation %s: Body %d too close to central mass -> removed\n", GSF[st].path, index_h[i + NBS] % def_MaxIndex);
							fprintf(logfile, "Body %d too close to central mass -> removed\n", index_h[i + NBS] % def_MaxIndex);
						}
						else{
							printf("In Simulation %s: Test Particle %d too close to central mass -> removed\n", GSF[st].path, index_h[i + NBS] % def_MaxIndex);
							fprintf(logfile, "Test Particle %d too close to central mass -> removed\n", index_h[i + NBS] % def_MaxIndex);
						}
					}
				}
				if(c < 0){
					if(Nst == 1) fprintf(ejectfile, "%.20g %d %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %d\n", time_h[0]/365.25, index_h[i + NBS], x4_h[i + NBS].w, v4_h[i + NBS].w, x4_h[i + NBS].x, x4_h[i + NBS].y, x4_h[i + NBS].z, v4_h[i + NBS].x, v4_h[i + NBS].y, v4_h[i + NBS].z, spin_h[i + NBS].x, spin_h[i + NBS].y, spin_h[i + NBS].z, c);
					else fprintf(ejectfile, "%.20g %d %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %d\n", time_h[st]/365.25, index_h[i + NBS] % def_MaxIndex, x4_h[i + NBS].w, v4_h[i + NBS].w, x4_h[i + NBS].x, x4_h[i + NBS].y, x4_h[i + NBS].z, v4_h[i + NBS].x, v4_h[i + NBS].y, v4_h[i + NBS].z, spin_h[i + NBS].x, spin_h[i + NBS].y, spin_h[i + NBS].z, c);
					
					EjectionEnergyCall(st, i);
				}
			}
			fclose(ejectfile);
			fclose(logfile);
		}
	}
}


//This function removes ghost particles and reorders the arrays
//It returns 1 if a simulation has less than the minimal number of bodies, otherwise zero
 int Data::remove(){
	int NminFlag = 0;
	NBmax = 0;
	for(int st = 0; st < Nst; ++st){
#if USE_NAF == 1
		remove_cpu /*1, 1*/ (x4_h, v4_h, a_h, N_h, Nsmall_h, index_h, spin_h, love_h, migration_h, createFlag_h, test_h, EnergySum_h, rcrit_h, rcritv_h, NBS_h[st], st, aelimits_h, aecount_h, enccount_h, aecountT_h, enccountT_h, K_h, Kold_h, NB[st], NconstT, P.SLevels, P.UseMigrationForce, P.CreateParticles, naf.x_h, naf.y_h, naf.n);
#else
		remove_cpu /*1, 1*/ (x4_h, v4_h, a_h, N_h, Nsmall_h, index_h, spin_h, love_h, migration_h, createFlag_h, test_h, EnergySum_h, rcrit_h, rcritv_h, NBS_h[st], st, aelimits_h, aecount_h, enccount_h, aecountT_h, enccountT_h, K_h, Kold_h, NB[st], NconstT, P.SLevels, P.UseMigrationForce, P.CreateParticles, NULL, NULL, 0);
#endif
		resize(N_h[st], NB[st], 1);
		resize(N_h[st] + Nsmall_h[st], NBT[st], 0);

		if(N_h[st] < Nmin[st].x){
			NminFlag = 1;
		}
		if(Nsmall_h[st] < Nmin[st].y){
			NminFlag = 2;
		}

	}
	return NminFlag;
}



// **************************************
//This function recomputes the value of NB, which is the next bigger 
//number to N which is a power of two.
//also called for NBT with test particles
 void Data::resize(int N, int &NB, int f){

	NB = 16;
	if( N > 16) NB = 32;
	if( N > 32) NB = 64;
	if( N > 64) NB = 128;
	if( N > 128) NB = 256;
	if( N > 256) NB = 512;
	if( N > 512) NB = 1024;
	if( N > 1024) NB = 2048;
	if( N > 2048) NB = 4096;
	if( N > 4096) NB = 8192;
	if( N > 8192) NB = 16384;
	if( N > 16384) NB = 32768;
	if( N > 32768) NB = 65536;
	if( N > 65536) NB = 131072;
	if( N > 131072) NB = 262144;

	if(f == 1){
		//avoid max
		if(NB > NBmax){
			NBmax = NB;
		}
	}
}


//This function rearranges the memory if a simulations is stopped
//It runs with only one thread on the GPU, to avoid unnecesary data copies
void removeM_cpu(double4 *x4_h, double4 *v4_h, double4 *xold_h, double4 *vold_h, double4 *spin_h, double3 *love_h, double3 *migration_h, int *createFlag_h, double3 *a_h, double *test_h, int *index_h, double *rcrit_h,
double *rcritv_h, int st, int NBS, int NsmallS, int *N_h, int *Nsmall_h, int NT, int NsmallT, const int NconstT, float4 *aelimits_h, unsigned int *aecount_h, unsigned int *enccount_h, unsigned long long *aecountT_h, unsigned long long *enccountT_h, const int SLevels, const int UseMigrationForce, const int CreateParticles, double *nafx_h, double *nafy_h, int nafn, int2 *Encpairs2_h, int Nh){

	for(int j = 0; j < N_h[st]; ++j){
//printf("removeM %d %d %d %d %d\n", st, N_h[st], j, j + NBS, j + NT);
		Encpairs2_h[j + NBS].x = j + NT;
		Encpairs2_h[j + NBS].y = Nh;
		x4_h[j + NT] = x4_h[j + NBS];
		v4_h[j + NT] = v4_h[j + NBS];
		xold_h[j + NT] = xold_h[j + NBS];
		vold_h[j + NT] = vold_h[j + NBS];
		spin_h[j + NT] = spin_h[j + NBS];
		love_h[j + NT] = love_h[j + NBS];
		if(UseMigrationForce > 0){
			migration_h[j + NT] = migration_h[j + NBS];
		}
		if(CreateParticles > 0){
			createFlag_h[j + NT] = createFlag_h[j + NBS];
		}
		a_h[j + NT] = a_h[j + NBS];
		test_h[j + NT] = test_h[j + NBS];
		index_h[j + NT] = index_h[j + NBS];
		for(int l = 0; l < SLevels; ++l){
			rcrit_h[j + NT + l * NconstT] = rcrit_h[j + NBS + l * NconstT];
			rcritv_h[j + NT + l * NconstT] = rcritv_h[j + NBS + l * NconstT];
		}
		aelimits_h[j + NT] = aelimits_h[j + NBS];
		enccount_h[j + NT] = enccount_h[j + NBS];
		aecount_h[j + NT] = aecount_h[j + NBS];
		aecountT_h[j + NT] = aecountT_h[j + NBS];
		enccountT_h[j + NT] = enccountT_h[j + NBS];
		for(int i = 0; i < nafn; ++i){
			nafx_h[(j + NT) * nafn + i] = nafx_h[(j + NBS) * nafn + i];
			nafy_h[(j + NT) * nafn + i] = nafy_h[(j + NBS) * nafn + i];
		}
	}
}


//this kernel rearranges the simulations index
void remove3M_cpu(int *index_h, int *N_h, int *NBS_h, const int Nst){

	int idy = 0;
	int st = 0;

	for(st = 0; st < Nst; ++st){

		int N = N_h[st];
		int NBS = NBS_h[st];

		for(idy = 0; idy < N; ++idy){

			int index = index_h[idy + NBS] % def_MaxIndex;
			index_h[idy + NBS] = index + st * def_MaxIndex;
	//printf("index %d %d %d\n", st, index + st * def_MaxIndex, N);
		}
	}
}


//this kernel rearranges the indices of the prechecker list
void remove4M_cpu(int2 *Encpairs_h, int2 *Encpairs2_h, const int Nencpairs){

	int id = 0 * 1 + 0;

	for(id = 0 * 1 + 0; id < Nencpairs; ++id){

		int i = Encpairs_h[id].x;
		int j = Encpairs_h[id].y;
		
		int ii = Encpairs2_h[i].x;
		int jj = Encpairs2_h[j].x;

		Encpairs_h[id].x = ii;
		Encpairs_h[id].y = jj;

		if(Encpairs2_h[i].y == 0) Encpairs_h[id].x = -1;
		if(Encpairs2_h[j].y == 0) Encpairs_h[id].y = -1;

	}
}



// This function stops simulations with less than the minimal number of bodies 
// or if the simulation has ended.
// it rearanges the memory
 void Data::stopSimulations(){
	NT = 0;
	NsmallT = 0;
	NEnergyT = 0;

	for(int st = 0; st < Nst; ++st){

		//In the following, set N_h to zero for all simulations which should be stopped
		if(N_h[st] < Nmin[st].x){
			N_h[st] = 0;
		}
		if(Nsmall_h[st] < Nmin[st].y){
			N_h[st] = 0;
		}

		if(StopFlag_m[0] > 0 && timeStep >= delta_h[st]){
			N_h[st] = 0;
		}
		if(P.StopAtEncounter > 0 && n1_h[st] < 0){
			N_h[st] = 0;

		}
		//rearange arrays//
#if USE_NAF == 1
		removeM_cpu /* 1, 1*/ (x4_h, v4_h, xold_h, vold_h, spin_h, love_h, migration_h, createFlag_h, a_h, test_h, index_h, rcrit_h, rcritv_h,  
					    st, NBS_h[st], NsmallS_h[st], N_h, Nsmall_h, NT, NsmallT, NconstT, aelimits_h,
					    aecount_h, enccount_h, aecountT_h, enccountT_h, P.SLevels, P.UseMigrationForce, P.CreateParticles, naf.x_h, naf.y_h, naf.n, Encpairs2_h, N_h[st]);
#else
		removeM_cpu /* 1, 1*/ (x4_h, v4_h, xold_h, vold_h, spin_h, love_h, migration_h, createFlag_h, a_h, test_h, index_h, rcrit_h, rcritv_h, 
					    st, NBS_h[st], NsmallS_h[st], N_h, Nsmall_h, NT, NsmallT, NconstT, aelimits_h,
					    aecount_h, enccount_h, aecountT_h, enccountT_h, P.SLevels, P.UseMigrationForce, P.CreateParticles, NULL, NULL, 0, Encpairs2_h, N_h[st]);
#endif

		NBS_h[st] = NT;
		NsmallS_h[st] = NsmallT;
		NEnergy[st] = NEnergyT;
		NT += N_h[st];
		NsmallT += Nsmall_h[st];
		NEnergyT += 8;
	}


	for(int st = 0; st < Nst; ++st){

//printf("stop simulations  %d %d %d %d Nst %d Ntot %d\n", st, N_h[st], Nmin[st].x, Nmin[st].y, Nst, NT);
		int s = 0;
		if(timeStep >= delta_h[st]){
			printf("In Simulation %s: Reached the end, simulation stopped\n", GSF[st].path);
			fprintf(masterfile,"In Simulation %s: Reached the end, simulation stopped\n", GSF[st].path);
			GSF[st].logfile = fopen(GSF[st].logfilename, "a");
			fprintf(GSF[st].logfile,"Reached the end, simulation stopped\n");
			fclose(GSF[st].logfile);
			s = 1;
		}
		else if(N_h[st] < Nmin[st].x){
			if(P.StopAtEncounter > 0 && n1_h[st] < 0){
				if(Nst > 1){
					printf("In Simulation %s: Close Encounter occurred, simulation stopped\n", GSF[st].path);
					fprintf(masterfile,"In Simulation %s: Close Encounter occurred, simulation stopped\n", GSF[st].path);
					GSF[st].logfile = fopen(GSF[st].logfilename, "a");
					fprintf(GSF[st].logfile,"Close Encounter occurred, simulation stopped\n");
					fclose(GSF[st].logfile);
					s = 1;
				}
				else{
					printf("Close Encounter occurred, simulation stopped\n");
					fprintf(masterfile,"Close Encounter occurred, simulation stopped\n");
					GSF[st].logfile = fopen(GSF[st].logfilename, "a");
					fprintf(GSF[st].logfile,"Close Encounter occurred, simulation stopped\n");
					fclose(GSF[st].logfile);
					s = 1;
				}
			}
			else{
				if(Nst > 1){
					if(Nsmall_h[st] < Nmin[st].y){
						printf("In Simulation %s: Number of test particles smaller than NminTP, simulation stopped\n", GSF[st].path);
						fprintf(masterfile,"In Simulation %s: Number of test particles smaller than NminTP, simulation stopped\n", GSF[st].path);
						GSF[st].logfile = fopen(GSF[st].logfilename, "a");
						fprintf(GSF[st].logfile,"Number of test particles smaller than NminTP, simulation stopped\n");
						fclose(GSF[st].logfile);
					}
					else{
						printf("In Simulation %s: Number of bodies smaller than Nmin, simulation stopped\n", GSF[st].path);
						fprintf(masterfile,"In Simulation %s: Number of bodies smaller than Nmin, simulation stopped\n", GSF[st].path);
						GSF[st].logfile = fopen(GSF[st].logfilename, "a");
						fprintf(GSF[st].logfile,"Number of bodies smaller than Nmin, simulation stopped\n");
						fclose(GSF[st].logfile);
					}
					s = 1;
				}
				else{
					if(Nsmall_h[st] < Nmin[st].y){
						printf("Number of test particles smaller than NminTP, simulation stopped\n");
						fprintf(masterfile,"Number of test particles smaller than NminTP, simulation stopped\n");
						GSF[0].logfile = fopen(GSF[0].logfilename, "a");
						fprintf(GSF[0].logfile,"Number of test particles smaller than NminTP, simulation stopped\n");
						fclose(GSF[0].logfile);
					}
					else{
						printf("Number of bodies smaller than Nmin, simulation stopped\n");
						fprintf(masterfile,"Number of bodies smaller than Nmin, simulation stopped\n");
						GSF[0].logfile = fopen(GSF[0].logfilename, "a");
						fprintf(GSF[0].logfile,"Number of bodies smaller than Nmin, simulation stopped\n");
						fclose(GSF[0].logfile);
					}
					s = 1;
				}
			}
		}
		if(s == 1){
			for(int sst = st; sst < Nst - 1; ++sst){
				GSF[sst] = GSF[sst + 1];

				NB[sst] = NB[sst + 1];
				Nmin[sst].x = Nmin[sst + 1].x;
				Nmin[sst].y = Nmin[sst + 1].y;
				rho[sst] = rho[sst + 1];
				n1_h[sst] = n1_h[sst + 1];
				n2_h[sst] = n2_h[sst + 1];
				N_h[sst] = N_h[sst + 1];
				Nsmall_h[sst] = Nsmall_h[sst + 1];
				Msun_h[sst] = Msun_h[sst + 1];
				Spinsun_h[sst] = Spinsun_h[sst + 1];
				Lovesun_h[sst] = Lovesun_h[sst + 1];
				J2_h[sst] = J2_h[sst + 1];
				idt_h[sst] = idt_h[sst + 1];
				ict_h[sst] = ict_h[sst + 1];
				Rcut_h[sst] = Rcut_h[sst + 1];
				RcutSun_h[sst] = RcutSun_h[sst + 1];
				time_h[sst] = time_h[sst + 1];
				dt_h[sst] = dt_h[sst + 1];
				delta_h[sst] = delta_h[sst + 1];

				U_h[sst] = U_h[sst + 1];
				LI_h[sst] = LI_h[sst + 1];
				Energy0_h[sst] = Energy0_h[sst + 1];
				LI0_h[sst] = LI0_h[sst + 1];

				NBS_h[sst] = NBS_h[sst + 1];
				NsmallS_h[sst] = NsmallS_h[sst + 1];
				NEnergy[sst] = NEnergy[sst + 1];

				for(int j = 0; j < 8; ++j){
					int NE0 = NEnergy[sst];
					int NE1 = NEnergy[sst + 1];
					Energy_h[NE0 + j] = Energy_h[NE1 + j];
				}
			}
			st -= 1;
			Nst -= 1;

		}
	}




	if(Nst > 0){
		remove3M_cpu /* Nst, NBmax */ (index_h, N_h, NBS_h, Nst);
		if(Nencpairs_h[0] > 0) remove4M_cpu /* (Nencpairs_h[0] + 255) / 256, 256 */ (Encpairs_h, Encpairs2_h, Nencpairs_h[0]);
	}
}

#if def_CPU == 1
 void Data::ElapsedTime(float *times, timeval tt1, timeval tt2){
	
	times[0] = (1000 * tt2.tv_sec + 0.001 * tt2.tv_usec - 1000 * tt1.tv_sec - 0.001 * tt1.tv_usec); //time in milliseconds
}
#endif

 int Data::freeOrbit(){
	
	int error;
	
	free(x4_h);
	free(v4_h);
	free(index_h);
	free(spin_h);
	free(love_h);
	if(P.UseMigrationForce > 0){
		free(migration_h);
	}
	if(P.CreateParticles > 0){
		free(createFlag_h);
	}
	free(rcrit_h);
	free(aelimits_h);
	free(aecount_h);
	free(enccount_h);
	free(aecountT_h);
	free(enccountT_h);

	free(coordinateBuffer_h);
	free(coordinateBufferIrr_h);
	free(timestepBuffer);
	free(timestepBufferIrr);
	free(NBuffer);
	free(NBufferIrr);

	free(RV_h);
	free(RVObs_h);
	free(TransitTime_h);
	free(TransitTimeObs_h);
	free(NtransitsT_h);
	free(NRVT_h);
	free(NtransitsTObs_h);
	free(NRVTObs_h);
	free(elementsA_h);
	free(elementsB_h);
	free(elementsT_h);
	free(elementsSpin_h);
	free(elementsL_h);
	free(elementsC_h);
	free(elementsP_h);
	free(elementsSA_h);
	free(elementsI_h);
	free(elementsM_h);
	free(elementsCOV_h);

	free(groupIterate_h);

	free(U_h);
	free(LI_h);
	free(Energy_h);
	free(Energy0_h);
	free(LI0_h);

#if def_CPU == 0

#else
#if def_CPU == 1
	free(xold_h);
	free(vold_h);
	free(x4b_h);
	free(v4b_h);
	free(x4bb_h);
	free(v4bb_h);
	free(a_h);
	free(b_h);
	free(ab_h);
	free(indexb_h);
	free(indexbb_h);
	free(rcritb_h);
	free(rcritbb_h);
	free(rcritv_h);
	free(rcritvb_h);
	free(rcritvbb_h);
	free(spinb_h);
	free(spinbb_h);
	free(vcom_h);
	free(EnergySum_h);
	free(Encpairs_h);
	free(Encpairs2_h);
	free(Encpairs3_h);
	free(scan_h);

	free(groupIndex_h);

	//BSA
	free(xt_h);
	free(vt_h);
	free(xp_h);
	free(vp_h);
	free(dx_h);
	free(dv_h);
	free(dt1_h);
	free(t1_h);
	free(dtgr_h);
	free(Coltime_h);
	free(BSstop_h);

	if(P.WriteEncounters == 2){
		free(morton_h);
		free(sortRank_h);
		free(sortCount_h);
		free(sortIndex_h);
		free(leafNodes_h);
		free(internalNodes_h);
	}
#endif


	free(Nenc_m);
	free(Ncoll_m);
	free(Ntransit_m);
	free(NWriteEnc_m);
	free(EjectionFlag_m);
	free(nFragments_m);
	free(EncFlag_m);
	free(StopFlag_m);
	free(ErrorFlag_m);
	free(test_h);
	free(Nencpairs_h);
	free(Nencpairs2_h);
	free(Nencpairs3_h);
#endif

	free(Coll_h);
	free(writeEnc_h);
	free(Fragments_h);

#if def_poincareFlag == 1
	free(PFlag_h);
#endif	
	free(BSAstop_h);

	if(P.UseMigrationForce > 0){
	}
	if(P.CreateParticles > 0){
	}


	if(P.WriteEncounters == 2){
	}

	if(Nst > 1){
	}

	if(P.ndev > 1){
	}
	if(P.ndev > 2){
	}
	if(P.ndev > 3){
	}





	
#if def_poincareFlag == 1
#endif
#if def_G3 > 0
#endif

#if USE_RANDOM == 1
#endif



	
	error = 0;
	if(error != 0){
		printf("Cuda Orbit free error = %d = %s\n",error, "-");
		fprintf(masterfile, "Cuda Orbit free error = %d = %s\n",error, "-");
		return 0;
	}
	return 1;
}

