// energy.hpp
// Energy functions for PFHub Benchmark 6 v2
// Questions/comments to trevor.keller@nist.gov (Trevor Keller)

#ifndef CAHNHILLIARD_ENERGY
#define CAHNHILLIARD_ENERGY
#include<cmath>

#define cid 0
#define uid 1
#define pid 2

// Composition parameters
const double w  = 5.0;  // well height
const double Ca = 0.30; // alpha composition
const double Cb = 0.70; // beta composition
const double C0 = 0.50; // system composition
const double C1 = 0.04; // fluctuation magnitude

// Electrostatic parameters
const double k       = 0.3;  // charge-neutralization factor
const double epsilon = 20.0; // permittivity

// Physical parameters
const double kappa = 2.0;
const double M0    = 10.0;

// Gauss-Seidel parameters
double tolerance = 1.0e-12;      // Choose wisely. 1e-10 is the minimum tolerance for which mass is conserved.
unsigned int residual_step = 10; // number of iterations between residual computations
unsigned int max_iter = 1000000; // don't let the solver stagnate
double omega = 1.2;              // relaxation parameter (default is 1.2): 1 is stock Gauss-Seidel, 1.2 is successive over-relaxation, 0.8 is successive under-relaxation.

// Energy equations
double cheminit(const double& x, const double& y)
{
	// Equation 12
	return C0 + C1 * ( std::cos(0.200 * x) * std::cos(0.110 * y)
	                 + std::pow(std::cos(0.130 * x) * std::cos(0.087 * y), 2.0)
	                 + std::cos(0.025 * x - 0.150*y)
	                 * std::cos(0.070 * x - 0.020 * y));
}

template<typename T>
double chemenergy(const T& C)
{
	// Equation 2
	const double A = C - Ca;
	const double B = Cb - C;
	return w * A*A * B*B;
}

template<typename T>
double elecenergy(const T& C, const T& P)
{
	// Equation 3
	const double rhoTot = k * (C - C0);
	return 0.5 * rhoTot * P;
}

template <int dim, typename T>
double pExt(const MMSP::grid<dim,MMSP::vector<T> >& GRID, const MMSP::vector<int>& x)
{
	const double A = 0.0002;
	const double B =-0.0100;
	const double C = 0.0200;
	const double hx = MMSP::dx(GRID);
	const double hy = MMSP::dy(GRID);

	return A * hx * x[0] * hy * x[1]
		 + B * hx * x[0]
	 	 + C * hy * x[1];
}

// Energy derivatives
template<typename T>
double dfchemdc(const T& C)
{
	// d(chemenergy)/dc
	const double A = C - Ca;
	const double B = Cb - C;
	return 2.0 * w * A * B * (Ca + Cb - 2.0 * C);
}

template<typename T>
double dfelecdc(const T& P)
{
	return 0.5 * k * (1. - C0) * P;
}

template<typename T>
double dfcontractivedc(const T& C, const T& Cnew)
{
	double nonlinearCoeff = 2.0 * w * (2.0 * C*C + Ca*Ca + 4.0*Ca*Cb + Cb*Cb);
	return nonlinearCoeff * Cnew;
}

template<typename T>
double dfexpansivedc(const T& C, const T& P)
{
	return - 2.0 * w * (3.0 * (Ca + Cb) * C*C + Ca * Cb * (Ca + Cb))
		   + 2.0 * dfelecdc(P);
}

// Discrete Laplacian operator missing the central value, for implicit source terms
template<int dim, typename T>
double fringe_laplacian(const MMSP::grid<dim,MMSP::vector<T> >& GRID, const MMSP::vector<int>& x, const int field);

#endif
