#include "Orbit2CPU.h"

#if def_CPU == 1
double CreateParticlesParameters_c[12];
#endif


// **************************************
// This kernel generates new test particles according to ranges in Kepler elements
//
// Date: September 2022
// Author: Simon Grimm
// **************************************
void create1_cpu(int *random_h, double4 *x4_h, double4 *v4_h, double4 *spin_h, double3 *love_h, int *index_h, const int NN, const double dt, const double Msun, const double time, int MaxIndex, double *Fragments_h, int *nFragments_m, const int NT){
#if USE_RANDOM == 1

	int id = 0 * 1 + 0;

	double p =  CreateParticlesParameters_c[1];		//probability of creating a new particle, per year
	double aa = CreateParticlesParameters_c[2];		//in AU
	double da = CreateParticlesParameters_c[3];		//in AU
	double ee = CreateParticlesParameters_c[4];
	double de = CreateParticlesParameters_c[5];
	double iinc = CreateParticlesParameters_c[6];
	double dinc = CreateParticlesParameters_c[7];
	double m =  CreateParticlesParameters_c[8];		//in Solar Masses
	double r =  CreateParticlesParameters_c[9];		//in AU

	//printf("%g %g %g %g %g %g %g %g %g\n", p, aa, da, ee, de, iinc, dinc, m, r);


	for(id = 0 * 1 + 0; id < NN; ++id){

		int random = random_h[id];
		double rd = drand48();
		random_h[id] = random;

		p = p / 365.25 * dt / dayUnit;          //probability per time step and per thread

		if(rd < p){

			double a = drand48() * da + aa - 0.5 * da;   //in AU
			double e = drand48() * de + ee - 0.5 * de;
			double inc = drand48() * dinc + iinc - 0.5 * dinc;

			if(inc < 0.0) inc = 2.0 * M_PI - inc;
			if(e < 0.0) e = 0.0;

			double Omega = drand48() * 2.0 * M_PI;
			double w = drand48() * 2.0 * M_PI;
			double M = drand48() * 2.0 * M_PI;


			double4 spin4 = {0.0, 0.0, 0.0, 0.4};
			double3 love3 = {0.0, 0.0, 0.0};

			double4 x4i, v4i;

			x4i.w = m;
			v4i.w = r;

			x4i.x = a;
			x4i.y = e;
			x4i.z = inc;
			v4i.x = Omega;
			v4i.y = w;
			v4i.z = M;
			KepToCart_M(x4i, v4i, Msun);

#if def_CPU == 0
			int nf = atomicAdd(&nFragments_m[0], 1);
#else
			int nf;
			#pragma omp atomic capture
			nf = nFragments_m[0]++;
#endif

			if(NN + nf < NT){
printf("Create particle, %d %d\n", nf, MaxIndex + nf + 1);
				x4_h[NN + nf] = x4i;
				v4_h[NN + nf] = v4i;
				index_h[NN + nf] = MaxIndex + nf + 1;
				spin_h[NN + nf] = spin4;
				love_h[NN + nf] = love3;

				Fragments_h[nf * 25 + 0] = time/365.25;
				Fragments_h[nf * 25 + 1] = (double)(MaxIndex + nf + 1);
				Fragments_h[nf * 25 + 2] = x4i.w;
				Fragments_h[nf * 25 + 3] = v4i.w;
				Fragments_h[nf * 25 + 4] = x4i.x;
				Fragments_h[nf * 25 + 5] = x4i.y;
				Fragments_h[nf * 25 + 6] = x4i.z;
				Fragments_h[nf * 25 + 7] = v4i.x;
				Fragments_h[nf * 25 + 8] = v4i.y;
				Fragments_h[nf * 25 + 9] = v4i.z;
				Fragments_h[nf * 25 + 10] = spin4.x;
				Fragments_h[nf * 25 + 11] = spin4.y;
				Fragments_h[nf * 25 + 12] = spin4.z;

			}
		}
	}
#endif
}

// **************************************
// This kernel generates new test particles around a parent body
//
// Date: September 2022
// Author: Simon Grimm
// **************************************
void create2_cpu(int *random_h, double4 *x4_h, double4 *v4_h, double4 *spin_h, double3 *love_h, int *createFlag_h, int *index_h, const int NN, const double dt, const double time, int MaxIndex, double *Fragments_h, int *nFragments_m, const int NT){
#if USE_RANDOM == 1

	int id = 0 * 1 + 0;

	double p = CreateParticlesParameters_c[1];		//probability of creating a new particle, per year
	double m = CreateParticlesParameters_c[8];		//mass in Solar Masses
	double r = CreateParticlesParameters_c[9];		//radius in AU
	double Vmin = CreateParticlesParameters_c[10];		//Velocity factor
	double Vmax = CreateParticlesParameters_c[11];		//Velocity factor

	//printf("%g %g %g %g %g %g %g %g %g\n", p, aa, da, ee, de, iinc, dinc, m, r);


	for(id = 0 * 1 + 0; id < NN; ++id){

		if(createFlag_h[id] == 1){
			int random = random_h[id];
			double rd = drand48();
			random_h[id] = random;

			p = p / 365.25 * dt / dayUnit;          //probability per time step and per thread

			if(rd < p){

				double4 xp = x4_h[id];		//Coordinates of the parent body
				double4 vp = v4_h[id];
				//velocits
		
				//escape velocity at 3 times the physical radius of the parent body
				double vesc = sqrt(2.0 * def_ksq * xp.w / (3.0 * vp.w));

				double v = (drand48() * (Vmax - Vmin) + Vmin) * vesc;

				//direction 
				double u = drand48();
				double theta = drand48() * 2.0 * M_PI;

				//sign
				double s = drand48();

				//move the new particle 3 times the physical radius of the parent body away
				double x = 3 * vp.w * sqrt(1.0 - u * u) * cos(theta);
				double y = 3 * vp.w * sqrt(1.0 - u * u) * sin(theta);
				double z = 3 * vp.w * u;

				volatile double vx = v * sqrt(1.0 - u * u) * cos(theta);
				volatile double vy = v * sqrt(1.0 - u * u) * sin(theta);
				volatile double vz = v * u;

				if( s > 0.5){
					z *= -1.0;
					vz *= -1.0;
				}


				double4 spin4 = {0.0, 0.0, 0.0, 0.4};
				double3 love3 = {0.0, 0.0, 0.0};

				double4 x4i, v4i;

				x4i.w = m;
				v4i.w = r;

				x4i.x = xp.x + x;
				x4i.y = xp.y + y;
				x4i.z = xp.z + z;
				v4i.x = vp.x + vx;
				v4i.y = vp.y + vy;
				v4i.z = vp.z + vz;


#if def_CPU == 0
				int nf = atomicAdd(&nFragments_m[0], 1);
#else
				int nf;
				#pragma omp atomic capture
				nf = nFragments_m[0]++;
#endif

				if(NN + nf < NT){
printf("Create particle, %d %d %d\n", id, nf, MaxIndex + nf + 1);
					x4_h[NN + nf] = x4i;
					v4_h[NN + nf] = v4i;
					index_h[NN + nf] = MaxIndex + nf + 1;
					spin_h[NN + nf] = spin4;
					love_h[NN + nf] = love3;
					createFlag_h[NN + nf] = 0;

					Fragments_h[nf * 25 + 0] = time/365.25;
					Fragments_h[nf * 25 + 1] = (double)(MaxIndex + nf + 1);
					Fragments_h[nf * 25 + 2] = x4i.w;
					Fragments_h[nf * 25 + 3] = v4i.w;
					Fragments_h[nf * 25 + 4] = x4i.x;
					Fragments_h[nf * 25 + 5] = x4i.y;
					Fragments_h[nf * 25 + 6] = x4i.z;
					Fragments_h[nf * 25 + 7] = v4i.x;
					Fragments_h[nf * 25 + 8] = v4i.y;
					Fragments_h[nf * 25 + 9] = v4i.z;
					Fragments_h[nf * 25 + 10] = spin4.x;
					Fragments_h[nf * 25 + 11] = spin4.y;
					Fragments_h[nf * 25 + 12] = spin4.z;
				}
			}
		}
	}
#endif
}



int Data::createCall(){
	nFragments_m[0] = 0;

	if(P.CreateParticles == 1){
		create1_cpu /* 1, 1 */ (random_h, x4_h, v4_h, spin_h, love_h, index_h, N_h[0] + Nsmall_h[0], dt_h[0], Msun_h[0].x, time_h[0], MaxIndex, Fragments_h, nFragments_m, P.CreateParticlesN);
	}

	if(P.CreateParticles == 2){
		create2_cpu /* (N_h[0] + Nsmall_h[0] + 127) / 128, 128 */ (random_h, x4_h, v4_h, spin_h, love_h, createFlag_h, index_h, N_h[0] + Nsmall_h[0], dt_h[0], time_h[0], MaxIndex, Fragments_h, nFragments_m, P.CreateParticlesN);
	}


	if(N_h[0] + Nsmall_h[0] + nFragments_m[0] > P.CreateParticlesN){
		nFragments_m[0] = P.CreateParticlesN - N_h[0] - Nsmall_h[0];
	}

	if(nFragments_m[0] > 0){
		Nsmall_h[0] += nFragments_m[0];
		MaxIndex += nFragments_m[0];
	}
	return 1;
}

//read total number of particles
int Data::createReadFile1(){

	printf("Read create Particles file 1: %s\n", P.CreateParticlesfilename);


	FILE *CreateParticlesfile;
	CreateParticlesfile = fopen(P.CreateParticlesfilename, "r");

	char sp[160];
	int er;

	P.CreateParticlesN = 0;

	for(int j = 0; j < 1000; ++j){ //loop around all lines in the param.dat file
		int c;
		for(int i = 0; i < 50; ++i){
			c = fgetc(CreateParticlesfile);
			if(c == EOF) break;
			sp[i] = char(c);
			if(c == '=' || c == ':'){
				sp[i + 1] = '\0';
				break;
			}
		}
		if(c == EOF) break;
		if(strcmp(sp, "Create Particles mode =") == 0){
			er = fscanf (CreateParticlesfile, "%d", &P.CreateParticles);
			if(er <= 0){
				printf("Error: Create Particles mode value not valid!\n");
				return 0;
			}
			if(fgets(sp, 3, CreateParticlesfile) != nullptr)
			continue;
		}

		if(strcmp(sp, "Maximum number of particles =") == 0){
			er = fscanf (CreateParticlesfile, "%d", &P.CreateParticlesN);
			if(er <= 0){
				printf("Error: Maximum number of particles value not valid!\n");
				return 0;
			}
			if(fgets(sp, 3, CreateParticlesfile) != nullptr)
			break;
		}
	}
	//avoid max
	if(P.CreateParticlesN > P.Nfragments){
		P.Nfragments = P.CreateParticlesN;
	}

	printf("NCreate %d %d\n", P.CreateParticlesN, P.Nfragments);

	fclose(CreateParticlesfile);

	return 1;
}

int Data::createReadFile2(){

	printf("Read create Particles file\n");
	printf("NconstT %d\n", NconstT);

	FILE *CreateParticlesfile;
	CreateParticlesfile = fopen(P.CreateParticlesfilename, "r");

	char sp[160];
	int er;

	double p = 0.0;
	double a = 1.0;
	double da = 0.0;
	double e = 0.0;
	double de = 0.0;
	double inc = 0.0;
	double dinc = 0.0;
	double m = 0.0;
	double r = 1.0e-10;
	double Vmin = 0.99;
	double Vmax = 1.0;

	for(int j = 0; j < 1000; ++j){ //loop around all lines in the param.dat file
		int c;
		for(int i = 0; i < 50; ++i){
			c = fgetc(CreateParticlesfile);
			if(c == EOF) break;
			sp[i] = char(c);
			if(c == '=' || c == ':'){
				sp[i + 1] = '\0';
				break;
			}
		}
		if(c == EOF) break;

		if(strcmp(sp, "Create Particles mode =") == 0){
			er = fscanf (CreateParticlesfile, "%d", &P.CreateParticles);
			if(er <= 0){
				printf("Error: Create Particles mode value not valid!\n");
				return 0;
			}
			if(fgets(sp, 3, CreateParticlesfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "Maximum number of particles =") == 0){
			int skip;
			er = fscanf (CreateParticlesfile, "%d", &skip);
			if(er <= 0){
				printf("Error: Maximum number of particles value not valid!\n");
				return 0;
			}
			if(fgets(sp, 3, CreateParticlesfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "Creation rate =") == 0){
			er = fscanf (CreateParticlesfile, "%lf", &p);
			if(er <= 0){
				printf("Error: Creation rate value not valid!\n");
				return 0;
			}
			if(fgets(sp, 3, CreateParticlesfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "a =") == 0){
			er = fscanf (CreateParticlesfile, "%lf", &a);
			if(er <= 0){
				printf("Error: a value not valid!\n");
				return 0;
			}
			if(fgets(sp, 3, CreateParticlesfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "da =") == 0){
			er = fscanf (CreateParticlesfile, "%lf", &da);
			if(er <= 0){
				printf("Error: da value not valid!\n");
				return 0;
			}
			if(fgets(sp, 3, CreateParticlesfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "e =") == 0){
			er = fscanf (CreateParticlesfile, "%lf", &e);
			if(er <= 0){
				printf("Error: e value not valid!\n");
				return 0;
			}
			if(fgets(sp, 3, CreateParticlesfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "de =") == 0){
			er = fscanf (CreateParticlesfile, "%lf", &de);
			if(er <= 0){
				printf("Error: de value not valid!\n");
				return 0;
			}
			if(fgets(sp, 3, CreateParticlesfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "inc =") == 0){
			er = fscanf (CreateParticlesfile, "%lf", &inc);
			if(er <= 0){
				printf("Error: inc value not valid!\n");
				return 0;
			}
			if(fgets(sp, 3, CreateParticlesfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "dinc =") == 0){
			er = fscanf (CreateParticlesfile, "%lf", &dinc);
			if(er <= 0){
				printf("Error: dinc value not valid!\n");
				return 0;
			}
			if(fgets(sp, 3, CreateParticlesfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "m =") == 0){
			er = fscanf (CreateParticlesfile, "%lf", &m);
			if(er <= 0){
				printf("Error: m value not valid!\n");
				return 0;
			}
			if(fgets(sp, 3, CreateParticlesfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "r =") == 0){
			er = fscanf (CreateParticlesfile, "%lf", &r);
			if(er <= 0){
				printf("Error: r value not valid!\n");
				return 0;
			}
			if(fgets(sp, 3, CreateParticlesfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "Vmin =") == 0){
			er = fscanf (CreateParticlesfile, "%lf", &Vmin);
			if(er <= 0){
				printf("Error: Vmin value not valid!\n");
				return 0;
			}
			if(fgets(sp, 3, CreateParticlesfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "Vmax =") == 0){
			er = fscanf (CreateParticlesfile, "%lf", &Vmax);
			if(er <= 0){
				printf("Error: Vmax value not valid!\n");
				return 0;
			}
			if(fgets(sp, 3, CreateParticlesfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "List of particle indizes:") == 0){
			if(fgets(sp, 3, CreateParticlesfile) != nullptr)
			for(int i = 0; i < N_h[0] + Nsmall_h[0]; ++i){	
				int id;
				er = fscanf (CreateParticlesfile, "%d", &id);

				if(er <= 0){
					break;
				}
				for(int j = 0; j < N_h[0] + Nsmall_h[0]; ++j){
					if(index_h[j] == id){
						createFlag_h[j] = 1;
printf("%d %d %d\n", i, id, j);
						break;
					}

				}
			}
			continue;
		}
		printf("Undefined line in param.dat file: line %d\n", j);
		return 0;

	}

	printf("%d\n", P.CreateParticles);

	fclose(CreateParticlesfile);

	if(P.CreateParticles == 2){
	}


	double parameters[12];
	parameters[0] = P.CreateParticles;
	parameters[1] = p;
	parameters[2] = a;
	parameters[3] = da;
	parameters[4] = e;
	parameters[5] = de;
	parameters[6] = inc;
	parameters[7] = dinc;
	parameters[8] = m;
	parameters[9] = r;
	parameters[10] = Vmin;
	parameters[11] = Vmax;

#if def_CPU == 0
	cudaMemcpyToSymbol(CreateParticlesParameters_c, parameters, 12 * sizeof(double), 0, cudaMemcpyHostToDevice);
#else
	memcpy(CreateParticlesParameters_c, parameters, 12 * sizeof(double));
#endif


	return 1;
}