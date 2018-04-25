#include <random>
#include <iostream>
#include <cmath>
#include <time.h>
#include <omp.h>
using namespace std;

const int N = 1000;
const float eps = 0.01f;
const int maxIteration = 500;
const int MAXRAND = 100;

float checkSumResult(float* x, float* xCheck, int N) {
	float sum = 0.0f;
	for (int i = 0; i < N; i++) {
		sum += fabs(x[i] - xCheck[i]);
	}
	return sum / (float)N;
}

float checkMaxResult(float* x, float* xCheck, int N) {
	float max = 0.0f;
	for (int i = 0; i < N; i++) {
		if (x[i] - xCheck[i] > max) {
			max = x[i] - xCheck[i];
		}
		if (x[i] - xCheck[i] < -max) {
			max = x[i] - xCheck[i];
		}
	}
	return max;
}

void multMatrixVector(float* matrix, float* x, float* b, int N) {
		float sum = 0;
#pragma omp parallel for shared (sum)
	for (int i = 0; i < N; i++) {
		sum = 0;
		for (int j = 0; j < N; j++) {
			sum += matrix[i*N + j] * x[j];
		}
		b[i] = sum;
	}
}

void printVector(int n, float* x) {
	for (int i = 0; i < n; i++) {
		printf("%f  ", x[i]);
	}
	printf("\n");
}

void printMatrix(int n, float *matrix) {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			printf("%f   ", matrix[n*i + j]);
		}
		printf("\n");
	}
	printf("\n");
}

bool checkMainDiag(float* matrix, int N) {
	float num = 0, sum = 0;
	for (int i = 0; i < N; i++) {
		num = fabs(matrix[i*N + i]);
		sum = 0.0f;
		for (int j = 0; j < N; j++) {
			if (i != j) {
				sum += fabs(matrix[i*N + j]);
			}
		}
		if (num > sum) {
			return false;
		}
	}
	return true;
}

//#pragma offload_attribute(push, target(mic))

void findX(float* matrix, float * x, float* tempX, float * b, int N) {
	for (int i = 0; i < N; i++) {
		x[i] = b[i];
		for (int j = 0; j < N; j++) {
			if (i != j) {
				x[i] -= matrix[i*N + j] * tempX[j];
			}
		}
		x[i] /= matrix[i*N + i];
	}
}

void findXVectorized(float* matrix, float * x, float* tempX, float * b, int N) {
	for (int i = 0; i < N; i++) {
		x[i] = b[i];
		for (int j = 0; j < N; j++) {
			if (i != j) {
				x[i] -= matrix[i*N + j] * tempX[j];
			}
		}
		x[i] /= matrix[i*N + i];
	}
}

void findXVectParallel(float* matrix, float * x, float* tempX, float * b, int N) {
#pragma omp parallel for 
	for (int i = 0; i < N; i++) {
		x[i] = b[i];
		for (int j = 0; j < N; j++) {
			if (i != j) {
				x[i] -= matrix[i*N + j] * tempX[j];
			}
		}
	}
#pragma omp parallel for
	for (int i = 0; i < N; i++) {
		x[i] /= matrix[i*N + i];
	}
}

void yakobi(float eps, float* matrix, float* b, float* x, int N, int *iteration) {
	float max = 0.0f;
	float * tempX = new float[N];
	do {
		(*iteration)++;
		findX(matrix, x, tempX, b, N);
		max = fabs(x[0] - tempX[0]);
		
#pragma omp parallel for shared (max)
		for (int i = 1; i < N; ++i) {
			if (max < fabs(x[i] - tempX[i])) {
				max = fabs(x[i] - tempX[i]);
			}
		}
		if (*(iteration) > maxIteration) {
			cout << "max iteration: " << (*iteration) << endl;
			return;
		}
		for (int i = 0; i < N; ++i) {
			tempX[i] = x[i];
		}
	} while (max > eps);
	delete[] tempX;
}

void yakobiVectorized(float eps, float* matrix, float* b, float* x, int N, int *iteration) {
	float max = 0;
	float * tempX = new float[N];
	do {
		(*iteration)++;
		findXVectorized(matrix, x, tempX, b, N);
		max = fabs(x[0] - tempX[0]);
#pragma omp parallel for shared(max)
		for (int i = 1; i < N; ++i) {
			if (max < fabs(x[i] - tempX[i])) {
				max = fabs(x[i] - tempX[i]);
			}
		}
		if (*(iteration) > maxIteration) {
			cout << "max iteration: " << (*iteration) << endl;
			return;
		}
		for (int i = 0; i < N; ++i) {
			tempX[i] = x[i];
		}
	} while (max > eps);
	delete[] tempX;
	return;
}

void yakobiVectParallel(float eps, float* matrix, float* b, float* x, int N, int *iteration) {
	float max = 0;
	float * tempX = new float[N];
	do {
		(*iteration)++;
		findXVectParallel(matrix, x, tempX, b, N);
		max = fabs(x[0] - tempX[0]);
#pragma omp parallel for shared(max)
		for (int i = 1; i < N; ++i) {
			if (max < fabs(x[i] - tempX[i])) {
				max = fabs(x[i] - tempX[i]);
			}
		}
		if (*(iteration) > maxIteration) {
			cout << "max iteration: " << (*iteration) << endl;
			return;
		}
#pragma omp parallel for
		for (int i = 0; i < N; ++i) {
			tempX[i] = x[i];
		}
	} while (max > eps);
	delete[] tempX;
	return;
}

//#pragma offload_attribute(pop)

int main() {

	float CPUTimeStart, MICTimeStart;
	float CPUTimeFinish, MICTimeFinish;

	float CPUTimeStartVect, MICTimeStartVect;
	float CPUTimeFinishVect, MICTimeFinishVect;

	float CPUTimeStartVectParallel, MICTimeStartVectParallel;
	float CPUTimeFinishVectParallel, MICTimeFinishVectParallel;

	float* matrix = new float[N*N];

	float* CPUx = new float[N];
	float* MICx = new float[N];
	float* CPUVectX = new float[N];
	float* MICVectX = new float[N];
	float* CPUVectParallelX = new float[N];
	float* MICVectParallelX = new float[N];

	float* xCheck = new float[N];
	float* b = new float[N];

	int CPUiteration = 0, CPUVectIteration = 0, CPUVectParallelIteration = 0;
	int MICiteration = 0, MICVectIteration = 0, MICVectParallelIteration = 0;

	for (int i = 0; i < N*N; i++)
		matrix[i] = (float)(rand() % MAXRAND);
	for (int i = 0; i < N; i++) {
		//for diagonal dominant
		matrix[i*N + i] += MAXRAND*N;
		xCheck[i] = float(i + 1);
	}
	multMatrixVector(matrix, xCheck, b, N);
	if (checkMainDiag(matrix, N)) {
		cout << "No diagonally dominant\n" << endl;
		return 1;
	}
	/// //////////////////////////////////////////////
	CPUTimeStart = clock();
	yakobi(eps, matrix, b, CPUx, N, &CPUiteration);
	CPUTimeFinish = clock();
	printf("CPU time (no vec, no parallel) : %f sec , iterarion: %d:\n",
		(CPUTimeFinish - CPUTimeStart) / (float)CLOCKS_PER_SEC, CPUiteration);
	/// ///////////////////////////////////////////////
	CPUTimeStartVect = clock();
	yakobiVectorized(eps, matrix, b, CPUVectX, N, &CPUVectIteration);
	CPUTimeFinishVect = clock();
	printf("CPU time (Vect, no parallel) : %f sec , iterarion: %d\n",
		(CPUTimeFinishVect - CPUTimeStartVect) / (float)CLOCKS_PER_SEC, CPUVectIteration);
	/// ///////////////////////////////////////////////
	CPUTimeStartVectParallel = clock();
	yakobiVectParallel(eps, matrix, b, CPUVectParallelX, N, &CPUVectParallelIteration);
	CPUTimeFinishVectParallel = clock();
	printf("CPU time (Vect, Parallel) : %f sec , iterarion: %d\n\n",
		(CPUTimeFinishVectParallel - CPUTimeStartVectParallel) / (float)CLOCKS_PER_SEC, CPUVectParallelIteration);
	/// ///////////////////////////////////////////////
//#pragma offload terget(mic) in (matrix[0:N*N], b[0:N], MICiteration, MICVectIteration, MICVectParallelIteration) out(MICx[0:N], MICVectX[0:N], MICVectParallelX[0:N], MICiteration, MICVectIteration, MICVectParallelIteration)
	{
		MICTimeStart = clock();
		yakobi(eps, matrix, b, MICx, N, &MICiteration);
		MICTimeFinish = clock();
		printf("MIC time (no vec, no parallel) : %f sec , iterarion: %d\n",
			(MICTimeFinish - MICTimeStart) / (float)CLOCKS_PER_SEC, MICiteration);
		/// ///////////////////////////////////////////////
		MICTimeStartVect = clock();
		yakobiVectorized(eps, matrix, b, MICVectX, N, &MICVectIteration);
		MICTimeFinishVect = clock();
		printf("MIC time (Vect, no parallel) : %f sec , iterarion: %d\n",
			(MICTimeFinishVect - MICTimeStartVect) / (float)CLOCKS_PER_SEC, MICVectIteration);
		/// ///////////////////////////////////////////////
		MICTimeStartVectParallel = clock();
		yakobiVectParallel(eps, matrix, b, MICVectParallelX, N, &MICVectParallelIteration);
		MICTimeFinishVectParallel = clock();
		printf("MIC time (Vect, Parallel) : %f sec , iterarion: %d\n\n",
			(MICTimeFinishVectParallel - MICTimeStartVectParallel) / (float)CLOCKS_PER_SEC, MICVectParallelIteration);
		/// ///////////////////////////////////////////////
	}

	float CPUerrorsSum = checkSumResult(CPUx, xCheck, N);
	float MICerrorsSum = checkSumResult(MICx, xCheck, N);

	float CPUerrorsMax = checkMaxResult(CPUx, xCheck, N);
	float MICerrorsMax = checkMaxResult(MICx, xCheck, N);

	float CPUerrorsSumVect = checkSumResult(CPUVectX, xCheck, N);
	float MICerrorsSumVect = checkSumResult(MICVectX, xCheck, N);

	float CPUerrorsMaxVect = checkMaxResult(CPUVectX, xCheck, N);
	float MICerrorsMaxVect = checkMaxResult(MICVectX, xCheck, N);

	float CPUerrorsSumVectParallel = checkSumResult(CPUVectParallelX, xCheck, N);
	float MICerrorsSumVectParallel = checkSumResult(MICVectParallelX, xCheck, N);

	float CPUerrorsMaxVectParallel = checkMaxResult(CPUVectParallelX, xCheck, N);
	float MICerrorsMaxVectParallel = checkMaxResult(MICVectParallelX, xCheck, N);

	printf("CPUerrorsSum: %f\n", CPUerrorsSum);
	printf("CPUerrorsMax: %f\n\n", CPUerrorsMax);

	printf("CPUerrorsSumVect: %f\n", CPUerrorsSumVect);
	printf("CPUerrorsMaxVect: %f\n\n", CPUerrorsMaxVect);

	printf("CPUerrorsSumVectParallel: %f\n", CPUerrorsSumVectParallel);
	printf("CPUerrorsMaxVectParallel: %f\n\n", CPUerrorsMaxVectParallel);

	printf("MICerrorsSum: %f\n", MICerrorsSum);
	printf("MICerrorsMax: %f\n\n", MICerrorsMax);

	printf("MICerrorsSumVect: %f\n", MICerrorsSumVect);
	printf("MICerrorsMaxVect: %f\n\n", MICerrorsMaxVect);

	printf("MICerrorsSumVectParallel: %f\n", MICerrorsSumVectParallel);
	printf("MICerrorsMaxVectParallel: %f\n\n", MICerrorsMaxVectParallel);
	

	delete[] matrix;
	delete[] CPUx;
	delete[] MICx;
	delete[] CPUVectX;
	delete[] MICVectX;
	delete[] CPUVectParallelX;
	delete[] MICVectParallelX;
	delete[] xCheck;
	delete[] b;
	
	return 0;
}