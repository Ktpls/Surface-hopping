
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "timer.h"

#include <stdio.h>
#include <helper_cuda.h>
#include <cassert>
#include <fstream>
#include <iostream>
#define _USE_MATH_DEFINES
#include <math.h>
#include <sstream>
#include <complex>

//#define NANCHECKING
//#define TIMECOSTCHECKING
#define _KERNEL_DETACHED

using namespace std;

#define Allign( x, n ) ( ( (x) + (n) - 1) & (~( (n) - 1) ) )
#define nens 500
#define threadsPerBlock 1024
#define blocksPerGrid Allign(nens, threadsPerBlock) / threadsPerBlock
#define D2P31M 2147483647.0
#define D2P31 2147483648.0

const double DELX = 0.10610, DELY = 0.13930;
const double DELPX = 0.50 / DELX, DELPY = 0.50 / DELY;

#define WriteDeviceMem(dst,val) \
do{\
	double temp=(val);\
	checkCudaErrors(cudaMemcpyToSymbol((dst), &(temp), sizeof(double)));\
}while (0)

//called name_host on host
#define __BOTH_VAR__(type,name,suf) \
__device__ type name suf; \
type name##_host suf;
#define __UploadBOTHVAR__(name) WriteDeviceMem((name),(name##_host));
#define __WriteUploadBOTHVAR__(name,v)\
do{\
	(name##_host)=(v);\
	__UploadBOTHVAR__(name);\
}while (0)

__device__ double		massx, massy, kx, ky, a, b, delta;
__BOTH_VAR__(double, dt, )
__BOTH_VAR__(double, gamma, )
__device__ double		x1, x2, x3;
__device__ double		delx, dely, delpx, delpy;
__BOTH_VAR__(double, h, )
__device__ double dseed[nens];
double dseed_host;
__device__ double xi[1], ep;

__device__ double		s1loc_frag[nens * nens], s2loc_frag[nens * nens], betaloc_frag[nens * nens];
__device__ int		idflag[nens];
__device__ double		x[nens], y[nens], px[nens], py[nens];
__device__ double		alp[nens], beta[nens];
__device__ double		s1loc[nens], s2loc[nens], betaloc[nens];
__device__ double		temp1d[nens];
__BOTH_VAR__(double, g00, )

__device__ double rho11_frag[nens];
__device__ double rho22_frag[nens];
__device__ double alptot_frag[nens];
__device__ double betot_frag[nens];
__device__ double ekin_frag[nens];
__device__ double e1_frag[nens];
__device__ double e2_frag[nens];
__device__ double ecoh_frag[nens];

__device__ double forcex(double x, double y, int iflag)
{
	if (iflag == 1)
		return(-kx * (x - x1));
	else
		return(-kx * (x - x2));
}

__device__ double forcey(double x, double y, int iflag)
{
	return(-ky * y);
}


__device__ double pot(double x, double y, int iflag)
{
	if (iflag == 1)
		return((kx * pow(x - x1, 2.0)) / 2.0 + (ky * pow(y, 2.0)) / 2.0);
	else
		return(delta + (kx * pow(x - x1, 2.0)) / 2.0 + (ky * pow(y, 2.0)) / 2.0);
}


__device__ double w(double x, double y)
{
	return(pot(x, y, 1) - pot(x, y, 0));
}


__device__ double v12(double x, double y)
{
	return(exp(-a * pow((x - x3), 2.0) - b * pow(y, 2.0)) * gamma * y);
}


__device__ double dv12x(double x, double y)
{
	return(-2.0 * a * exp(-a * pow(x - x3, 2.0) - b * pow(y, 2.0)) * gamma * (x - x3) * y);
}


__device__ double dv12y(double x, double y)
{
	return(exp(-a * pow((x - x3), 2.0) - b * pow(y, 2.0)) * gamma -
		-2.0 * b * exp(-a * pow((x - x3), 2.0) - b * pow(y, 2.0)) * gamma * pow(y, 2.0));
}


__device__ void ggubs(double& DSEED, int NR, double* R)
{
	for (int I = 0; I < NR; I++)
	{
		DSEED = fmod(16807.0 * DSEED, D2P31M);
		R[I] = DSEED / D2P31;
		/* 5 R(I) = DSEED / D2P31 */
	}
}

void ggubs_host(double& DSEED, int NR, double* R)
{
	for (int I = 0; I < NR; I++)
	{
		DSEED = fmod(16807.0 * DSEED, D2P31M);
		R[I] = DSEED / D2P31;
		/* 5 R(I) = DSEED / D2P31 */
	}
}


void gaussrnd(double& dseed, double& grnd)
{
	double xi[1]; /* real*8 xi(1)? */
	grnd = 0;
	for (int n = 1; n <= 12; n++)
	{
		ggubs_host(dseed, 1, xi);
		grnd = grnd + xi[0];
	}
	grnd = grnd - 6;
}


__device__ double g(double x, double y, double px, double py)
{
	double arg = pow((x / (h * delx)), 2.0) + pow((y / (h * dely)), 2.0)
		+ pow((px / (h * delpx)), 2.0) + pow((py / (h * delpy)), 2.0);
	return exp(-0.5 * arg) / (4.0 * pow(M_PI, 2.0) * delx * dely * delpx * delpy * pow(h, 4.0));
}
double g_host(double x, double y, double px, double py)
{
	double arg = pow((x / (h_host * DELX)), 2.0) + pow((y / (h_host * DELY)), 2.0)
		+ pow((px / (h_host * DELPX)), 2.0) + pow((py / (h_host * DELPY)), 2.0);
	return exp(-0.5 * arg) / (4.0 * pow(M_PI, 2.0) * DELX * DELY * DELPX * DELPY * pow(h_host, 4.0));
}


void initialrnd(double xo, double yo, double pxo, double pyo, double& dseed)
{
	int	icount = 0;
	double	z;
	double		temp[nens];
	for (size_t n = 0; n < nens; n++)
	{
		gaussrnd(dseed, z);
		temp[n] = xo + DELX * z;
	}
	checkCudaErrors(cudaMemcpyToSymbol(x, temp, nens * sizeof(double)));

	for (size_t n = 0; n < nens; n++)
	{
		gaussrnd(dseed, z);
		temp[n] = yo + DELY * z;
	}
	checkCudaErrors(cudaMemcpyToSymbol(y, temp, nens * sizeof(double)));

	for (size_t n = 0; n < nens; n++)
	{
		gaussrnd(dseed, z);
		temp[n] = pxo + DELPX * z;
	}
	checkCudaErrors(cudaMemcpyToSymbol(px, temp, nens * sizeof(double)));

	for (size_t n = 0; n < nens; n++)
	{
		gaussrnd(dseed, z);
		temp[n] = pyo + DELPY * z;
	}
	checkCudaErrors(cudaMemcpyToSymbol(py, temp, nens * sizeof(double)));

	icount = nens;
	cout << "ensemble size generated=" << icount << endl;
}

#define complex16	std::complex < double >
#define TRAVERSAL	for ( int x = 1; x <= 2; x++ ) for ( int y = 1; y <= 2; y++ )
//index starts with 1 in mat
struct comp16mat
{
public:
	complex16 comp[2][2];
	comp16mat()
	{
	}
	comp16mat(const complex16& b)
	{
		(*this) = b;
	}
	comp16mat(comp16mat& b)
	{
		(*this) = b;
	}
	complex16& operator() (size_t x, size_t y)
	{
		return(comp[x - 1][y - 1]);
	}
	void operator = (const complex16& b)
	{
		TRAVERSAL
		(*this)(x, y) = b;
	}
	void operator = (comp16mat& b)
	{
		TRAVERSAL
		(*this)(x, y) = b(x, y);
	}
	comp16mat operator + (comp16mat& b)
	{
		comp16mat temp;
		TRAVERSAL
			temp(x, y) = (*this)(x, y) + b(x, y);
		return(temp);
	}
	comp16mat operator - (comp16mat& b)
	{
		comp16mat temp;
		TRAVERSAL
			temp(x, y) = (*this)(x, y) - b(x, y);


		return(temp);
	}
	comp16mat operator*(comp16mat& b)
	{
		comp16mat temp;
		TRAVERSAL
		{
			temp(x, y) = 0;
			for (int i = 1; i <= 2; i++)
				temp(x, y) += (*this)(i, y) * b(x, i);
		}
		return(temp);
	}
	friend comp16mat operator*(double a, comp16mat& b);
};

comp16mat operator*(double a, comp16mat& b)
{
	comp16mat temp;
	TRAVERSAL
		temp(x, y) = a * b(x, y);
	return(temp);
}
#undef TRAVERSAL

void cal_phase(
	double& rho11, double& rho22, double& alpha, double& beta,
	comp16mat& rho, comp16mat& one, comp16mat& b0, comp16mat& rho0,
	double& phase)
{
	comp16mat drho, rho1, b2, b1, c;
	complex16 trace;

	rho1(1, 1) = complex16(rho11, 0.0);
	rho1(2, 2) = complex16(rho22, 0.0);
	rho1(2, 1) = complex16(alpha, beta);
	rho1(1, 2) = complex16(alpha, -beta);
	/* rho1(1,2) = conjg(rho1(2,1)) */

	drho = rho1 - rho;
	rho = rho1;
	b2 = one + drho + 0.5 * (drho * drho);
	b1 = b0 * b2;
	c = b1 * rho0;
	b0 = b1;
	trace = c(1, 1) + c(2, 2);
	phase = atan2(trace.imag(), trace.real());

#ifdef NANCHECKING
	if (isnan(phase))
	{
		cout << "nan detected" << endl;
#ifdef _WIN32
		DebugBreak();
#else
		asm("int 3");
#endif
	}
#endif
}


double readDouble(fstream& fs)
{
	const size_t	BUFSIZE = 256;
	char		buf[BUFSIZE];
	fs.getline(buf, BUFSIZE);
	return(atof(buf));
}


__global__ void sum_kernel(double* v, int dim)
{
	int n = blockIdx.x * blockDim.x + threadIdx.x;
	if (n >= dim) return;

	for (int i = 1; i < dim; i <<= 1)
	{
		if (n % (i << 1) == i)
		{
			v[n - i] += v[n];
		}
		__syncthreads();
	}
}

double* getDevPtr(const void* dd)
{
	double* devptr;
	cudaGetSymbolAddress((void**)&devptr, dd);
	return devptr;
}

__device__ double sum(double* v, int dim)
{
	sum_kernel << <blocksPerGrid, threadsPerBlock >> > (v, dim);
	return v[0];
}

double sum_host(double* v, int dim)
{
	sum_kernel << <blocksPerGrid, threadsPerBlock >> > (v, dim);
	double ret;
	checkCudaErrors(cudaMemcpy(&ret, v, sizeof(double), cudaMemcpyDeviceToHost));
	return ret;
}

#define CUDAPRELOGUE1D int n = blockIdx.x * blockDim.x + threadIdx.x; if (n >= nens) return;

//for debugging
__device__ bool nancheck[nens] = { 0 };

void NanCheck()
{
	bool nancheck_host[nens];
	cudaMemcpyFromSymbol(nancheck_host, nancheck, nens * sizeof(bool));
	for (size_t i = 0; i < nens; i++)
	{
		if (nancheck_host[i])
		{
			printf("paranancheck true in %zd\n", i);
#ifdef _WIN32
			DebugBreak();
#else
			asm("int 3");
#endif
			//break;
		}
	}
}



__global__ void  cal_f4_ave(double* cal_f4_ave_ret)
{
	CUDAPRELOGUE1D
	if (n != 0)
		return;
	cal_f4_ave_ret[0] = sum(rho11_frag, nens) / nens;
	cal_f4_ave_ret[1] = sum(rho22_frag, nens) / nens;
	cal_f4_ave_ret[2] = sum(alptot_frag, nens) / nens;
	cal_f4_ave_ret[3] = sum(betot_frag, nens) / nens;
	cal_f4_ave_ret[4] = sum(ekin_frag, nens) / nens;
	cal_f4_ave_ret[5] = sum(e1_frag, nens) / nens;
	cal_f4_ave_ret[6] = sum(e2_frag, nens) / nens;
	cal_f4_ave_ret[7] = sum(ecoh_frag, nens) / nens;
}

#ifdef _KERNEL_DETACHED

__global__ void cal_f1_frag()
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= nens * nens)
		return;

	int n, k;
	n = idx / nens;
	k = idx % nens;

	double gfac = g(x[n] - x[k], y[n] - y[k], px[n] - px[k], py[n] - py[k]);
	s1loc_frag[n * nens + k] = idflag[k] * gfac;
	s2loc_frag[n * nens + k] = (1 - idflag[k]) * gfac;
	betaloc_frag[n * nens + k] = beta[k] * gfac;
}

__global__ void cal_f1()
{
	CUDAPRELOGUE1D
		s1loc[n] = sum(&s1loc_frag[n * nens], nens);
	s2loc[n] = sum(&s2loc_frag[n * nens], nens);
	betaloc[n] = sum(&betaloc_frag[n * nens], nens);

	s1loc[n] = (s1loc[n] - g00 * idflag[n]) / (1.0 * nens);
	s2loc[n] = (s2loc[n] - g00 * (1 - idflag[n])) / (1.0 * nens);
	betaloc[n] = betaloc[n] / (1.0 * nens);

#ifdef NANCHECKING
	nancheck[n] |= isnan(s1loc[n]);
	nancheck[n] |= isnan(s2loc[n]);
	nancheck[n] |= isnan(betaloc[n]);
#endif
}

__global__ void cal_f2()
{
	CUDAPRELOGUE1D
		double		da11, da22, dia12;
	double		term3;
	term3 = 2.0 * v12(x[n], y[n]) * betaloc[n] * dt;
	da11 = -term3 / (s1loc[n] + ep);
	da22 = term3 / (s2loc[n] + ep);
	dia12 = v12(x[n], y[n]) * dt * (2.0 * idflag[n] - 1.0);

	ggubs(dseed[n], 1, xi);

	if (idflag[n] == 1)
	{
		if (da11 < 0.0)
		{
			if (xi[0] < abs(da11))
				idflag[n] = 0;
		}
	}
	else if (idflag[n] == 0)
	{
		if (da22 < 0.0)
		{
			if (xi[0] < abs(da22))
				idflag[n] = 1;
		}
	}
	double atemp, btemp;
	atemp = alp[n] + w(x[n], y[n]) * beta[n] * dt;
	btemp = beta[n] + dia12 - w(x[n], y[n]) * alp[n] * dt;
	alp[n] = atemp;
	beta[n] = btemp;


#ifdef NANCHECKING
	nancheck[n] |= isnan(alp[n]);
	nancheck[n] |= isnan(beta[n]);
#endif
}

__global__ void  cal_f3()
{
	CUDAPRELOGUE1D

		double fdxold, fdxnew, fdyold, fdynew;
	fdxold = forcex(x[n], y[n], idflag[n]);
	fdyold = forcey(x[n], y[n], idflag[n]);
	x[n] = x[n] + px[n] * dt / massx + 0.50 * fdxold * dt * dt / massx;
	y[n] = y[n] + py[n] * dt / massy + 0.50 * fdyold * dt * dt / massy;
	fdxnew = forcex(x[n], y[n], idflag[n]);
	fdynew = forcey(x[n], y[n], idflag[n]);
	px[n] = px[n] + 0.50 * (fdxold + fdxnew) * dt;
	px[n] = px[n] - 2.0 * dv12x(x[n], y[n]) * alp[n] * dt;
	py[n] = py[n] + 0.50 * (fdyold + fdynew) * dt;
	py[n] = py[n] - 2.0 * dv12y(x[n], y[n]) * alp[n] * dt;
}

__global__ void  cal_f4_frag()
{
	CUDAPRELOGUE1D

		rho11_frag[n] = 1.0 * idflag[n];
	rho22_frag[n] = 1.0 * (1 - idflag[n]);
	alptot_frag[n] = alp[n];
	betot_frag[n] = beta[n];
	ekin_frag[n] = pow(px[n], 2.0) / (2.0 * massx) + pow(py[n], 2.0) / (2.0 * massy);
	e1_frag[n] = idflag[n] * pot(x[n], y[n], 1);
	e2_frag[n] = (1 - idflag[n]) * pot(x[n], y[n], 0);
	ecoh_frag[n] = 2.0 * v12(x[n], y[n]) * alp[n];


#ifdef NANCHECKING
	nancheck[n] |= isnan(rho11_frag[n]);
	nancheck[n] |= isnan(rho22_frag[n]);
	nancheck[n] |= isnan(alptot_frag[n]);
	nancheck[n] |= isnan(betot_frag[n]);
	nancheck[n] |= isnan(ekin_frag[n]);
	nancheck[n] |= isnan(e1_frag[n]);
	nancheck[n] |= isnan(e2_frag[n]);
	nancheck[n] |= isnan(ecoh_frag[n]);
#endif
}
#else //_KERNEL_DETACHED

__global__ void cal_f1_frag(int i)
{
	CUDAPRELOGUE1D
		double gfac = g(x[i] - x[n], y[i] - y[n], px[i] - px[n], py[i] - py[n]);
	s1loc_frag[i * nens + n] = idflag[n] * gfac;
	s2loc_frag[i * nens + n] = (1 - idflag[n]) * gfac;
	betaloc_frag[i * nens + n] = beta[n] * gfac;
}

__global__ void cal_f123()
{
	//for 1
	CUDAPRELOGUE1D
	cal_f1_frag << <blocksPerGrid, threadsPerBlock >> > (n);
	s1loc[n] = sum(&s1loc_frag[n * nens], nens);
	s2loc[n] = sum(&s2loc_frag[n * nens], nens);
	betaloc[n] = sum(&betaloc_frag[n * nens], nens);

	s1loc[n] = (s1loc[n] - g00 * idflag[n]) / (1.0 * nens);
	s2loc[n] = (s2loc[n] - g00 * (1 - idflag[n])) / (1.0 * nens);
	betaloc[n] = betaloc[n] / (1.0 * nens);
#ifdef NANCHECKING
	nancheck[n] |= isnan(s1loc[n]);
	nancheck[n] |= isnan(s2loc[n]);
	nancheck[n] |= isnan(betaloc[n]);
#endif
	__syncthreads();


	//for2
	{
		double		da11, da22, dia12;
		double		term3;
		term3 = 2.0 * v12(x[n], y[n]) * betaloc[n] * dt;
		da11 = -term3 / (s1loc[n] + ep);
		da22 = term3 / (s2loc[n] + ep);
		dia12 = v12(x[n], y[n]) * dt * (2.0 * idflag[n] - 1.0);

		ggubs(dseed[n], 1, xi);

		if (idflag[n] == 1)
		{
			if (da11 < 0.0)
			{
				if (xi[0] < abs(da11))
					idflag[n] = 0;
			}
		}
		else if (idflag[n] == 0)
		{
			if (da22 < 0.0)
			{
				if (xi[0] < abs(da22))
					idflag[n] = 1;
			}
		}
		double atemp, btemp;
		atemp = alp[n] + w(x[n], y[n]) * beta[n] * dt;
		btemp = beta[n] + dia12 - w(x[n], y[n]) * alp[n] * dt;
		alp[n] = atemp;
		beta[n] = btemp;
#ifdef NANCHECKING
		nancheck[n] |= isnan(alp[n]);
		nancheck[n] |= isnan(beta[n]);
#endif
	}

	//for 3
	{
		double fdxold, fdxnew, fdyold, fdynew;
		fdxold = forcex(x[n], y[n], idflag[n]);
		fdyold = forcey(x[n], y[n], idflag[n]);
		x[n] = x[n] + px[n] * dt / massx + 0.50 * fdxold * dt * dt / massx;
		y[n] = y[n] + py[n] * dt / massy + 0.50 * fdyold * dt * dt / massy;
		fdxnew = forcex(x[n], y[n], idflag[n]);
		fdynew = forcey(x[n], y[n], idflag[n]);
		px[n] = px[n] + 0.50 * (fdxold + fdxnew) * dt;
		px[n] = px[n] - 2.0 * dv12x(x[n], y[n]) * alp[n] * dt;
		py[n] = py[n] + 0.50 * (fdyold + fdynew) * dt;
		py[n] = py[n] - 2.0 * dv12y(x[n], y[n]) * alp[n] * dt;
	}
}


__global__ void  cal_f4_frag()
{
	CUDAPRELOGUE1D

	rho11_frag[n] = 1.0 * idflag[n];
	rho22_frag[n] = 1.0 * (1 - idflag[n]);
	alptot_frag[n] = alp[n];
	betot_frag[n] = beta[n];
	ekin_frag[n] = pow(px[n], 2.0) / (2.0 * massx) + pow(py[n], 2.0) / (2.0 * massy);
	e1_frag[n] = idflag[n] * pot(x[n], y[n], 1);
	e2_frag[n] = (1 - idflag[n]) * pot(x[n], y[n], 0);
	ecoh_frag[n] = 2.0 * v12(x[n], y[n]) * alp[n];


#ifdef NANCHECKING
	nancheck[n] |= isnan(rho11_frag[n]);
	nancheck[n] |= isnan(rho22_frag[n]);
	nancheck[n] |= isnan(alptot_frag[n]);
	nancheck[n] |= isnan(betot_frag[n]);
	nancheck[n] |= isnan(ekin_frag[n]);
	nancheck[n] |= isnan(e1_frag[n]);
	nancheck[n] |= isnan(e2_frag[n]);
	nancheck[n] |= isnan(ecoh_frag[n]);
#endif

}
#endif //_KERNEL_DETACHED

void checkCUDACallErr(const char* const file,int const line) {
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		fprintf(stderr, "CUDA error at %s:%d code=%d(%s)\n", file, line, err, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

#define CHECKCUDACALLERR checkCUDACallErr(__FILE__, __LINE__)

int main(int argc, char** argv)
{
	{
		int dev = 0;
		cudaDeviceProp devProp;
		cudaError_t cudaStatus;
		cudaStatus = cudaGetDeviceCount(&dev);
		for (int i = 0; i < dev; i++) {
			cudaGetDeviceProperties(&devProp, i);
			cout << "使用GPU device " << dev << ": " << devProp.name << endl;
			cout << "SM的数量：" << devProp.multiProcessorCount << endl;
			cout << "每个线程块的共享内存大小：" << devProp.sharedMemPerBlock / 1024.0 << " KB" << endl;
			cout << "每个线程块的最大线程数：" << devProp.maxThreadsPerBlock << endl;
		}
	}

	int		iseed;
	double	t, xo, yo, pxo, pyo;
	double	rho11, rho22, alptot, betot;
	double	tconv, tsout;

	int	n, nstep, istep, k;
	int	icount, kcount, kskip;


	double		ekin, e1, e2, epot, ecoh, ediag, etot;
	double		tmax;
	double		phase;
	comp16mat	rho, one, b0, rho0;

	WriteDeviceMem(kx, 0.020);
	WriteDeviceMem(ky, 0.10);
	WriteDeviceMem(x1, 4.0);
	WriteDeviceMem(x2, 3.0);
	WriteDeviceMem(x3, 3.0);
	WriteDeviceMem(massx, 20000.0);
	WriteDeviceMem(massy, 6667.0);
	WriteDeviceMem(a, 3.0);
	WriteDeviceMem(b, 1.50);
	WriteDeviceMem(delta, 0.010);
	WriteDeviceMem(delx, DELX);
	WriteDeviceMem(dely, DELY);
	WriteDeviceMem(delpx, DELPX);
	WriteDeviceMem(delpy, DELPY);

	fstream f7("paramlvc.in", ios::in);
	fstream f15("mfhopd.out", ios::out);
	fstream f16("mfenergyd.out", ios::out);

	__WriteUploadBOTHVAR__(dt, readDouble(f7));
	iseed = readDouble(f7);
	xo = readDouble(f7);
	yo = readDouble(f7);
	pxo = readDouble(f7);
	pyo = readDouble(f7);
	tmax = readDouble(f7);
	__WriteUploadBOTHVAR__(gamma, readDouble(f7));

	f7.close();

	h_host = pow(4.0 / (6.0 * nens), 1.0 / 8.0);
	cout << "h=" << h_host << endl;
	__UploadBOTHVAR__(h);

	tconv = 0.02418880;
	tsout = 1.0 / tconv;

	kskip = int(tsout / dt_host);

	cout << "dt=" << dt_host << endl;
	cout << "iseed=" << iseed << endl;
	cout << "xo=" << xo << endl;
	cout << "yo=" << yo << endl;
	cout << "pxo=" << pxo << endl;
	cout << "pyo=" << pyo << endl;
	cout << "gamma=" << gamma_host << endl;

	nstep = int(tmax / dt_host);
	cout << "nstep=" << nstep << endl;
	cout << "kskip=" << kskip << endl;

#define EP 1e-10
	WriteDeviceMem(ep, EP);
	cout << "ep =" << EP << endl;



	rho11 = 0.0;
	rho22 = 0.0;
	alptot = 0.0;
	betot = 0.0;
	ekin = 0.0;
	e1 = 0.0;
	e2 = 0.0;
	ecoh = 0.0;
	ediag = 0.0;

	phase = 0.0;
	one = complex16(0.0, 0.0);
	one(1, 1) = complex16(1.0, 0.0);
	one(2, 2) = complex16(1.0, 0.0);
	rho0 = complex16(0.0, 0.0);
	b0 = one;

	dseed_host = 1.0 * iseed;
	initialrnd(xo, yo, pxo, pyo, dseed_host);
	{
		double		dseed_array_host[nens];
		for (int n = 0; n < nens; n++)
		{
			double xi[1];
			ggubs_host(dseed_host, 1, xi);
			dseed_array_host[n] = dseed_host;
		}
		checkCudaErrors(cudaMemcpyToSymbol(dseed, dseed_array_host, nens * sizeof(double)));
	}

	{
		int idflag_host[nens];
		double alp_host[nens], beta_host[nens];
		for (int n = 0; n < nens; n++)
		{
			idflag_host[n] = 1;
			alp_host[n] = 0.0;
			beta_host[n] = 0.0;
		}
		checkCudaErrors(cudaMemcpyToSymbol(idflag, idflag_host, nens * sizeof(int)));
		checkCudaErrors(cudaMemcpyToSymbol(alp, alp_host, nens * sizeof(double)));
		checkCudaErrors(cudaMemcpyToSymbol(beta, beta_host, nens * sizeof(double)));
	}

	char outputbuf[256];
#define WRITE_ALL(t) \
do{\
	int len;\
	len=sprintf(outputbuf,"%10.4lf  %16.10lf  %16.10lf  %16.10lf  %16.10lf  %16.10lf  \n",(t),rho11,rho22,alptot,betot,phase);\
	f15.write(outputbuf, len);\
	f15.flush();\
	len = sprintf(outputbuf, "%10.4lf  %16.10lf  %16.10lf  %16.10lf  %16.10lf  %16.10lf  \n", (t), etot, ekin, epot, ediag, ecoh);\
	f16.write(outputbuf, len);\
	f16.flush();\
}while (0)

	//buf to pass back result
	double* cal_f4_ave_ret;//on dev
	double cal_f4_result[8];//on host
	checkCudaErrors(cudaMalloc((void**)&cal_f4_ave_ret, 8 * sizeof(double)));

	cal_f4_frag << <blocksPerGrid, threadsPerBlock >> > ();
	cudaDeviceSynchronize();
	cal_f4_ave << <1, 1 >> > (cal_f4_ave_ret);

	checkCudaErrors(cudaMemcpy(cal_f4_result, cal_f4_ave_ret, 8 * sizeof(double), cudaMemcpyDeviceToHost));
	rho11 = cal_f4_result[0];
	rho22 = cal_f4_result[1];
	alptot = cal_f4_result[2];
	ekin = cal_f4_result[3];
	e1 = cal_f4_result[4];
	e2 = cal_f4_result[5];
	ecoh = cal_f4_result[6];
	epot = e1 + e2;
	ediag = ekin + e1 + e2;
	etot = ediag + ecoh;
	cal_phase(rho11, rho22, alptot, betot, rho, one, b0, rho0, phase);
	WRITE_ALL(0.0);

	icount = 0;
	kcount = 0;

	__WriteUploadBOTHVAR__(g00, g_host(0.0, 0.0, 0.0, 0.0));
	cout << "g00=" << g00_host << endl;

	double timeusage = 0;
	int tucount = 0;

	timer tmr(TIMER_STARTNOW);
#ifdef TIMECOSTCHECKING
	timer tmr_pf(TIMER_STARTNOW);
#endif
	for (istep = 1; istep <= nstep; istep++)
	{

		icount = icount + 1;
		kcount = kcount + 1;
		t = istep * dt_host;

#ifdef TIMECOSTCHECKING
		tmr_pf.restart();
#endif

#ifdef _KERNEL_DETACHED
		cal_f1_frag << <Allign(nens * nens, threadsPerBlock) / threadsPerBlock, threadsPerBlock >> > ();
		//cudaDeviceSynchronize();
		cal_f1 << <blocksPerGrid, threadsPerBlock >> > ();
		cal_f2 << <blocksPerGrid, threadsPerBlock >> > ();
		cal_f3 << <blocksPerGrid, threadsPerBlock >> > ();
		cal_f4_frag << <blocksPerGrid, threadsPerBlock >> > ();
		//cudaDeviceSynchronize();
		cal_f4_ave << <1, 1 >> > (cal_f4_ave_ret);
#else
		cal_f123 << <blocksPerGrid, threadsPerBlock >> > ();
		cal_f4_frag << <blocksPerGrid, threadsPerBlock >> > ();
#endif

		checkCudaErrors(cudaMemcpy(cal_f4_result, cal_f4_ave_ret, 8 * sizeof(double), cudaMemcpyDeviceToHost));
		rho11 = cal_f4_result[0];
		rho22 = cal_f4_result[1];
		alptot = cal_f4_result[2];
		ekin = cal_f4_result[3];
		e1 = cal_f4_result[4];
		e2 = cal_f4_result[5];
		ecoh = cal_f4_result[6];

		epot = e1 + e2;
		ediag = ekin + e1 + e2;
		etot = ediag + ecoh;
		cal_phase(rho11, rho22, alptot, betot, rho, one, b0, rho0, phase);

#ifdef NANCHECKING
		NanCheck();
#endif
		if (kcount == kskip)
		{
			WRITE_ALL(t * tconv);
			cout << int(t * tconv + 0.50) << ", " << rho11 << ", " << rho22 << ", " << rho11 + rho22 << ", " << phase << endl;
			kcount = 0;
		}
		//if (icount % 100 == 0)
		//	printf("%d\n", icount);
#ifdef TIMECOSTCHECKING
		timeusage += tmr_pf.gettime();
		tucount++;
#endif
	}
	cout <<"time cost total: "<< tmr.gettime() << endl;

#ifdef TIMECOSTCHECKING
	cout << timeusage/tucount << endl;
#endif

	cudaFree(cal_f4_ave_ret);
	cudaDeviceReset();
	system("pause");
}