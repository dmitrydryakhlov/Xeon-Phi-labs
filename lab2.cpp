#include <iostream>
#include <cmath>
#include <time.h>
#include <omp.h>
using namespace std;

const int N = 1024;

float checkSumResult(float* C, float* CheckC, int N) {
	float sum = 0.0f;
	for (int i = 0; i < N*N; i++) {
		sum += fabs(C[i] - CheckC[i]);
	}
	return sum / (float)(N*N);
}

float checkMaxResult(float* C, float* CheckC, int N) {
	float max = 0.0f;
	for (int i = 0; i < N*N; i++) {
		if (C[i] - CheckC[i] > max) {
			max = C[i] - CheckC[i];
		}
		if (C[i] - CheckC[i] < -max) {
			max = C[i] - CheckC[i];
		}
	}
	return max;
}

//#pragma offload_attribute(push, target(mic))
void matrixMult(float* A, float*  B, float*  C, int n) {
	float sum = 0;
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			sum = 0;
			//#pragma novector
			for (int k = 0; k < n; k++) {
				sum += A[i*n + k] * B[k*n + j];
			}
			C[n*i + j] = sum;
		}
	}
}

void matrixMultVect(float* __restrict A, float*  __restrict B, float*  __restrict C, int n) {
	float sum = 0;
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			sum = 0;
#pragma ivdep
#pragma omp simd reduction (+:sum)
#pragma vector allways
			for (int k = 0; k < n; k++) {
				sum += A[i*n + k] * B[k*n + j];
			}
			C[n*i + j] = sum;
		}
	}
}

void matrixMultVectParallel(float* __restrict A,  float* __restrict B, float* __restrict C, int n) {
	float sum = 0;
	for (int i = 0; i < n; i++) {
#pragma omp parallel for private(sum) 
		for (int j = 0; j < n; j++) {
			sum = 0;
#pragma ivdep
#pragma omp simd reduction (+:sum)
#pragma vector allways
			for (int k = 0; k < n; k++) {
				sum += A[i*n + k] * B[k*n + j];
			}
			C[n*i + j] = sum;
		}
	}
}

void printMatrix(int n, float *matrix) {
	printf("\n");
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			printf("%f   ", matrix[n*i + j]);
		}
		printf("\n");
	}
	printf("\n\n");
}
//#pragma offload_attribute(pop)


int main() {

	float CPUTimeStart, MICTimeStart;
	float CPUTimeFinish, MICTimeFinish;

	float CPUTimeStartVect, MICTimeStartVect;
	float CPUTimeFinishVect, MICTimeFinishVect;

	float CPUTimeStartVectParallel, MICTimeStartVectParallel;
	float CPUTimeFinishVectParallel, MICTimeFinishVectParallel;

	float* A = new float[N*N];
	float* B = new float[N*N];

	float* CPUC = new float[N*N];
	float* MICC = new float[N*N];
	float* CPUVectC = new float[N*N];
	float* MICVectC = new float[N*N];
	float* CPUVectParallelC = new float[N*N];
	float* MICVectParallelC = new float[N*N];
	for (int i = 0; i < N*N; ++i)
		A[i] = B[i] = (float)i;
	/// //////////////////////////////////////////////
	CPUTimeStart = clock();
	matrixMult(A, B, CPUC, N);
	CPUTimeFinish = clock();
	printf("CPU time (no vec, no parallel) : %f sec\n",
		(CPUTimeFinish - CPUTimeStart) / (float)CLOCKS_PER_SEC);
	/// ///////////////////////////////////////////////
	CPUTimeStartVect = clock();
	matrixMultVect(A, B, CPUVectC, N);
	CPUTimeFinishVect = clock();
	printf("CPU time (Vect, no parallel) : %f sec\n",
		(CPUTimeFinishVect - CPUTimeStartVect) / (float)CLOCKS_PER_SEC);
	/// ///////////////////////////////////////////////
	CPUTimeStartVectParallel = clock();
	matrixMultVectParallel(A, B, CPUVectParallelC, N);
	CPUTimeFinishVectParallel = clock();
	printf("CPU time (Vect, Parallel) : %f sec\n\n",
		(CPUTimeFinishVectParallel - CPUTimeStartVectParallel) / (float)CLOCKS_PER_SEC);
	/// ///////////////////////////////////////////////
	//#pragma offload target(mic) in (A[0:N*N], B[0:N*N]) out(MICC[0:N*N], MICVectC[0:N*N], MICVectParallelC[0:N*N])
	{
		MICTimeStart = clock();
		matrixMult(A, B, MICC, N);
		MICTimeFinish = clock();
		printf("MIC time (no vec, no parallel) : %f sec\n",
			(MICTimeFinish - MICTimeStart) / (float)CLOCKS_PER_SEC);
		/// ///////////////////////////////////////////////
		MICTimeStartVect = clock();
		matrixMultVect(A, B, MICVectC, N);
		MICTimeFinishVect = clock();
		printf("MIC time (Vect, no parallel) : %f sec\n",
			(MICTimeFinishVect - MICTimeStartVect) / (float)CLOCKS_PER_SEC);
		/// ///////////////////////////////////////////////
		MICTimeStartVectParallel = clock();
		matrixMultVectParallel(A, B, MICVectParallelC, N);
		MICTimeFinishVectParallel = clock();
		printf("MIC time (Vect, Parallel) : %f sec\n\n",
			(MICTimeFinishVectParallel - MICTimeStartVectParallel) / (float)CLOCKS_PER_SEC);
		/// ///////////////////////////////////////////////
	}

	float errorsSum = checkSumResult(CPUC, MICC, N);
	float errorsMax = checkMaxResult(CPUC, MICC, N);

	float errorsSumVect = checkSumResult(CPUVectC, CPUC, N);
	float errorsMaxVect = checkMaxResult(CPUVectC, MICC, N);

	float errorsSumVectParallel = checkSumResult(MICC, MICVectParallelC, N);
	float errorsMaxVectParallel = checkMaxResult(CPUC, MICVectParallelC, N);

	printf("errorsSum: %f\n", errorsSum);
	printf("errorsMax: %f\n\n", errorsMax);

	printf("errorsSumVect: %f\n", errorsSumVect);
	printf("errorsMaxVect: %f\n\n", errorsMaxVect);

	printf("errorsSumVectParallel: %f\n", errorsSumVectParallel);
	printf("errorsMaxVectParallel: %f\n\n", errorsMaxVectParallel);

	/*printMatrix(N, A);
	printMatrix(N, B);

	printMatrix(N, CPUC);
	printMatrix(N, MICC);

	printMatrix(N, CPUVectC);
	printMatrix(N, MICVectC);

	printMatrix(N, CPUVectParallelC);
	printMatrix(N, MICVectParallelC);*/

	delete[] A;
	delete[] B;
	delete[] CPUC;
	delete[] MICC;
	delete[] CPUVectC;
	delete[] MICVectC;
	delete[] MICVectParallelC;
	delete[] CPUVectParallelC;

	return 0;
}