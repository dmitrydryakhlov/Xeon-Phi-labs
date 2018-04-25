#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

void printVector(int n, float* vec) {
	for (int i = 0; i < n; i++) {
		printf("%f  ", vec[i]);
	}
	printf("\n");
}

float checkSumResult(int n, float* a, float* b) {
	float sum = 0.0;
	for (int i = 0; i < n; i++) {
		if (a[i] - b[i] < 0.0) {
			sum -= (a[i] - b[i]);
		}
		else {
			sum += (a[i] - b[i]);
		}
	}
	return sum / (float)n;
}

float checkMaxResult(int n, float* a, float* b) {
	float max = 0.0;
	for (int i = 0; i < n; i++) {
		if (a[i] - b[i] > max) {
			max = a[i] - b[i];
		}
		if (a[i] - b[i] < -max) {
			max = b[i] - a[i];
		}
	}
	return max;
}


void printMatrix(int n, float *matr) {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			printf("%f   ", matr[n*i + j]);
		}
		printf("\n");
	}
	printf("\n");
}

//#pragma offload_attribute(push, target(mic))
void dot(float* _Matrix, float* b, int n, float* result) {
#pragma omp parallel for
	for (int i = 0; i < n; i++) {
		float s = 0.0;
		for (int j = 0; j < n; j++) {
			s += _Matrix[i*n + j] * b[j];
		}
		result[i] = s;
	}
	return;
}
//#pragma offload_attribute(pop)

int main()
{
	int n = 10000;

	float CPUTimeStart;
	float CPUTimeFinish;
	float MICTimeStart;
	float MICTimeFinish;
	float* Matrix = new float[n*n];
	for (int i = 0; i < n*n; i++) {
		Matrix[i] = (float)rand() / RAND_MAX;
	}
	float* x = new float[n];
	float* res_cpu = new float[n],
		*res_mic = new float[n];
	for (int i = 0; i < n; ++i)
	{
		x[i] = (float)rand() / RAND_MAX;
	}
	CPUTimeStart = clock();
	dot(Matrix, x, n, res_cpu);
	CPUTimeFinish = clock();
	printf(" CPU time : %f :\n", (CPUTimeFinish - CPUTimeStart) / (float)CLOCKS_PER_SEC);

	//#pragma offload target(mic) in(Matrix[0:n*n], x[0:n]) out(res_mic[0:n])
	{
		MICTimeStart = clock();
		dot(Matrix, x, n, res_mic);
		MICTimeFinish = clock();
		printf(" MIC time : %f :\n", (MICTimeFinish - MICTimeStart) / (float)CLOCKS_PER_SEC);
	}
	float errorsSum = checkSumResult(n, res_cpu, res_mic);
	float errorsMax = checkMaxResult(n, res_cpu, res_mic);
	printf("errorsSum:%f\n", errorsSum);
	printf("errorsMax:%f\n", errorsMax);
	delete[] Matrix;
	delete[] res_cpu;
	delete[] res_mic;
	delete[] x;
	return 0;
}