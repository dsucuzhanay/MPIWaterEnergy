#include <stdio.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <mpi.h>

#define LENGTH 80

const int maxnum=100000;
double r[maxnum][3][3], rcutsq=1.44, L;
// r(number of molecule, atom 0=O,1=H,2=H, coordinate 0=x,1=y,2=z)

double sqr(double a){ return a*a; }

double energy12(int i1,int i2)
{
	int m, n, xyz;
	double shift[3], dr[3], mn[3], r6, distsq, dist, ene=0;
	const double sig=0.3166, eps=0.65, eps0=8.85e-12, e=1.602e-19, Na=6.022e23, q[3]={-0.8476,0.4238,0.4238};
	double elst, sig6;
	elst = e*e/(4*3.141593*eps0*1e-9)*Na/1e3,sig6=pow(sig,6);
	
	// periodic boundary conditions
	for (xyz=0; xyz<=2; xyz++)
	{
		dr[xyz] = r[i1][0][xyz]-r[i2][0][xyz];
		shift[xyz] = -L*floor(dr[xyz]/L+.5);	// round dr[xyz]/L to nearest integer
		dr[xyz] = dr[xyz]+shift[xyz];
	}
	
	distsq = sqr(dr[0])+sqr(dr[1])+sqr(dr[2]);
	if (distsq < rcutsq)
	{
		// calculate energy if within cutoff
		r6 = sig6/pow(distsq,3);
		ene = 4*eps*r6*(r6-1.);	// LJ energy
		for (m=0; m<=2; m++)
		{
			for (n=0; n<=2; n++)
			{
				for (xyz=0; xyz<=2; xyz++)
					mn[xyz] = r[i1][m][xyz]-r[i2][n][xyz]+shift[xyz];
				dist = sqrt(sqr(mn[0])+sqr(mn[1])+sqr(mn[2]));
				ene = ene+elst*q[m]*q[n]/dist;
			}
		}
	}
	
	return ene;
}

main(int argc, char *argv[])
{
	int me, nproc;
	int i, j, natoms, nmol, npair=0, totalPairs=0, maxnpair=0, minnpair=0;
	double energy=0, totalEnergy=0;
	FILE *fp;
	char line[LENGTH], nothing[LENGTH], name[20];
	
	// variables for timing
	clock_t cputime1, cputime2, cputime3;
	struct timeval time1, time2, time3;
	double dtimeReading, dtimeCalculation, dtimeTotalJob;
	float cpuTimeReading, cpuTimeCalculation, cpuTotalJob;
	float minCpuTimeReading, minCpuTimeCalculation, minCpuTotalJob, maxCpuTimeReading, maxCpuTimeCalculation, maxCpuTotalJob;
	double mindTimeReading, mindTimeCalculation, mindTimeTotalJob, maxdTimeReading, maxdTimeCalculation, maxdTimeTotalJob;
	
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &nproc);
	MPI_Comm_rank(MPI_COMM_WORLD, &me);
	
	// timing 1
	// timing is taken from this point because the name of the file is not entered by keyboard
	// and we need to measure the time of reading in all the threads
	cputime1 = clock();
	gettimeofday(&time1, NULL);
	
	// Option of reading 3
	if (me == 0)
	{
		printf("Program to calculate energy of water\n");
		printf("Input NAME of configuration file\n");
		scanf("%s", name);			// reading of filename from keyboard
		
		// timing 1
		//cputime1 = clock();
		//gettimeofday(&time1, NULL);
		
		fp = fopen(name, "r");		// opening of file and beginning of reading from HDD
		fgets(line, LENGTH, fp);	// skip first line
		fgets(line, LENGTH, fp);
		sscanf(line, "%i", &natoms);
		nmol = natoms/3;
		printf("Number of molecules %i\n", nmol);
		
		for (i=0; i<nmol; i++)
		{
			for(j=0; j<=2; j++)
			{
				fgets(line, LENGTH, fp);
				sscanf(line, "%s %s %s %lf %lf %lf",nothing,nothing,nothing, &r[i][j][0],&r[i][j][1],&r[i][j][2]);
			}
		}
		
		printf("First line: %lf %lf %lf\n",r[0][0][0],r[0][0][1],r[0][0][2]);
		fscanf(fp, "%lf", &L);			// read box size
		printf("Box size: %lf\n\n", L);
		
		fclose(fp);
	}
	
	// broadcast
	MPI_Bcast(&nmol, 1, 		 MPI_INT, 	 0, MPI_COMM_WORLD);
	MPI_Bcast(&r, 	 (nmol*3*3), MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&L, 	 1, 		 MPI_DOUBLE, 0, MPI_COMM_WORLD);
	
	// timing 2
	cputime2 = clock();
	gettimeofday(&time2, NULL);
	
	// time of reading
	//if (me == 0)
	//{
		cpuTimeReading = (float) (cputime2 - cputime1)/CLOCKS_PER_SEC;
		dtimeReading = ((time2.tv_sec - time1.tv_sec)+(time2.tv_usec - time1.tv_usec)/1e6);
	//}
	
	// parallelization
	for(i=me; i<nmol-1; i=i+nproc)
	{
		// calculate energy as sum over all pairs
		for(j=i+1; j<nmol; j++)
		{
			energy = energy + energy12(i,j);
			npair = npair + 1;
		}
	}
	
	printf("Process %i: number pairs = %i total energy = %lf \n", me, npair, energy);
	
	// total energy and total npairs
	MPI_Reduce(&energy, &totalEnergy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Reduce(&npair, 	&totalPairs,  1, MPI_INT, 	 MPI_SUM, 0, MPI_COMM_WORLD);
	
	// max and min npairs
	MPI_Reduce(&npair, &minnpair, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
	MPI_Reduce(&npair, &maxnpair, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
	
	// timing 3
	cputime3 = clock();
	gettimeofday(&time3, NULL);
	
	// time of calculation
	cpuTimeCalculation = (float) (cputime3 - cputime2)/CLOCKS_PER_SEC;
	dtimeCalculation = ((time3.tv_sec - time2.tv_sec)+(time3.tv_usec - time2.tv_usec)/1e6);
	
	// time total job
	//if (me == 0)
	//{
		cpuTotalJob = (float) (cputime3 - cputime1)/CLOCKS_PER_SEC;
		dtimeTotalJob = ((time3.tv_sec - time1.tv_sec)+(time3.tv_usec - time1.tv_usec)/1e6);
	//}
	
	// min timing
	MPI_Reduce(&cpuTimeReading, 	&minCpuTimeReading, 	1, MPI_FLOAT,  MPI_MIN, 0, MPI_COMM_WORLD);
	MPI_Reduce(&cpuTimeCalculation, &minCpuTimeCalculation, 1, MPI_FLOAT,  MPI_MIN, 0, MPI_COMM_WORLD);
	MPI_Reduce(&cpuTotalJob, 		&minCpuTotalJob, 		1, MPI_FLOAT,  MPI_MIN, 0, MPI_COMM_WORLD);
	MPI_Reduce(&dtimeReading, 		&mindTimeReading, 		1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
	MPI_Reduce(&dtimeCalculation, 	&mindTimeCalculation, 	1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
	MPI_Reduce(&dtimeTotalJob, 		&mindTimeTotalJob, 		1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
	
	// max timing
	MPI_Reduce(&cpuTimeReading, 	&maxCpuTimeReading, 	1, MPI_FLOAT,  MPI_MAX, 0, MPI_COMM_WORLD);
	MPI_Reduce(&cpuTimeCalculation, &maxCpuTimeCalculation, 1, MPI_FLOAT,  MPI_MAX, 0, MPI_COMM_WORLD);
	MPI_Reduce(&cpuTotalJob, 		&maxCpuTotalJob, 		1, MPI_FLOAT,  MPI_MAX, 0, MPI_COMM_WORLD);
	MPI_Reduce(&dtimeReading, 		&maxdTimeReading, 		1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	MPI_Reduce(&dtimeCalculation, 	&maxdTimeCalculation, 	1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	MPI_Reduce(&dtimeTotalJob, 		&maxdTimeTotalJob, 		1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	
	if (me == 0)
	{
		printf("\nTotal energy: %lf\n", 			totalEnergy);
		printf("Energy per molecule: %lf\n", 		totalEnergy/nmol);
		printf("Total pairs: %i\n", 				totalPairs);
		printf("Total pairs should be: %i\n", 		nmol*(nmol-1)/2);
		printf("Min npair: %i\n", 					minnpair);
		printf("Max npair: %i\n", 					maxnpair);
		printf("Load imbalance: %lf\n", 			2.*(maxnpair-minnpair)/(0.+maxnpair+minnpair));
		printf("Min cpu time reading: %lf\n", 		minCpuTimeReading);
		printf("Max cpu time reading: %lf\n", 		maxCpuTimeReading);
		printf("Min cpu time calculation: %lf\n",	minCpuTimeCalculation);
		printf("Max cpu time calculation: %lf\n",	maxCpuTimeCalculation);
		printf("Min cpu time total job: %lf\n", 	minCpuTotalJob);
		printf("Max cpu time total job: %lf\n", 	maxCpuTotalJob);
		printf("Min wall time reading: %lf\n",		mindTimeReading);
		printf("Max wall time reading: %lf\n",		maxdTimeReading);
		printf("Min wall time calculation: %lf\n",	mindTimeCalculation);
		printf("Max wall time calculation: %lf\n",	maxdTimeCalculation);
		printf("Min wall time total job: %lf\n",	mindTimeTotalJob);
		printf("Max wall time total job: %lf\n",	maxdTimeTotalJob);
	}
	
	MPI_Finalize();
}