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
const double w  = 5.00;  // well height
const double Ca = 0.30; // alpha composition
const double Cb = 0.70; // beta composition
const double Cs = 0.50; // system composition
const double Cf = 0.04; // fluctuation magnitude

// Electrostatic parameters
const double k       = 0.3; // charge-neutralization factor
const double epsilon = 20.; // permittivity
const double pA = 2.e-4;    // external field coefficients
const double pB =-1.e-2;
const double pC = 2.e-2;

// Physical parameters
const double kappa = 2.0;
const double M0    = 10.;

// Gauss-Seidel parameters
const double tolerance = 1.e-9;         // threshold residual ||b - Ax||₂ required to end iteration
const unsigned int residual_step = 100; // number of iterations between residual computations
const unsigned int max_iter = 100000;   // don't let the solver stagnate
const double omega = 1.2;               // relaxation parameter (default is 1.2):
                                        // omega = 1.0 is stock Gauss-Seidel,
                                        // omega = 1.2 is successive over-relaxation.

// Energy equations
double cheminit(const double& x, const double& y)
{
	// Equation 12
	return Cs + Cf * ( std::cos(0.200 * x) * std::cos(0.110 * y)
	                 + std::pow(std::cos(0.130 * x) * std::cos(0.087 * y), 2.0)
	                 + std::cos(0.025 * x - 0.150 * y)
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

double pExt(const double& xx, const double& yy)
{
	return pA * xx * yy
	     + pB * xx
	     + pC * yy;
}

template<typename T>
double elecenergy(const T& C, const T& C0, const T& P, const double& xx, const double& yy)
{
	// Equation 3
	const double rhoTot = k * (C - C0);
	return 0.5 * rhoTot * P + rhoTot * pExt(xx, yy);
}

template<typename T>
double dfcontractivedc(const T& C, const T& Cnew)
{
	double nonlinearCoeff = 2.0 * w * (2.0 * C*C + Ca*Ca + 4.0*Ca*Cb + Cb*Cb);
	return nonlinearCoeff * Cnew;
}

template<typename T>
double dfexpansivedc(const T& C)
{
	return - 2. * w * (Ca + Cb) * (3. * C*C + Ca * Cb);
}

// Discrete Laplacian operator missing the central value, for implicit source terms
template<int dim, typename T>
double fringe_laplacian(const MMSP::grid<dim,MMSP::vector<T> >& GRID, const MMSP::vector<int>& x, const int field);

#endif
