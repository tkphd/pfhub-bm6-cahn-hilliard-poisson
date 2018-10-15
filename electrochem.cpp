// electrochem.cpp
// Implementation of PFHub Benchmark 6 v2 using MMSP
// Questions/comments to trevor.keller@nist.gov (Trevor Keller)

#ifndef CAHNHILLIARDPOISSON_UPDATE
#define CAHNHILLIARDPOISSON_UPDATE
#include "MMSP.hpp"
#include <cmath>
#include "electrochem.hpp"
#include "energy.hpp"

namespace MMSP {

/*
	Field 0 is [c] composition.
	Field 1 is [u] chemical potential.
	Field 2 is [p] electrostatic potential.
*/

// Numerical parameters
const double deltaX = 0.5;
const double dt = 0.096 * (2.0 * std::pow(deltaX, 4) / (24.0 * M0 * kappa));
const double CFL = (24.0 * M0 * kappa * dt) / (2.0 * std::pow(deltaX, 4));

template<int dim, typename T>
double system_composition(const grid<dim,vector<T> >& GRID)
{
	// compute system composition
	double c0 = 0.0;
	double gridSize = static_cast<double>(nodes(GRID));
	#ifdef MPI_VERSION
	double localGridSize = gridSize;
	MPI::COMM_WORLD.Allreduce(&localGridSize, &gridSize, 1, MPI_DOUBLE, MPI_SUM);
	#endif

	#ifdef _OPENMP
	#pragma omp parallel for reduction(+:c0)
	#endif
	for (int n=0; n<nodes(GRID); n++)
		c0 += GRID(n)[cid];

	#ifdef MPI_VERSION
	double myC(c0);
	MPI::COMM_WORLD.Allreduce(&myC, &c0, 1, MPI_DOUBLE, MPI_SUM);
	#endif
	c0 /= gridSize;
	return c0;
}

template <int dim,typename T>
double Helmholtz(const grid<dim,vector<T> >& GRID, const T& C0)
{
	double dV = 1.0;
	for (int d=0; d<dim; d++)
		dV *= dx(GRID, d);

	double fchem = 0.0;
	double felec = 0.0;
	double fgrad = 0.0;

	#ifdef _OPENMP
	#pragma omp parallel for
	#endif
	for (int n=0; n<nodes(GRID); n++) {
		vector<int> x = position(GRID, n);
		const double xx = dx(GRID,0) * x[0];
		const double yy = dx(GRID,1) * x[1];
		vector<double> gradc = gradient(GRID, x, cid);

		double mygrad = gradc * gradc;
		double mychem = chemenergy(GRID(n)[cid]); // Equation 2
		double myelec = elecenergy(GRID(n)[cid], C0, GRID(n)[pid], xx, yy); // Equation 3

		#ifdef _OPENMP
		#pragma omp atomic
		#endif
		fgrad += mygrad;

		#ifdef _OPENMP
		#pragma omp atomic
		#endif
		fchem += mychem;

		#ifdef _OPENMP
		#pragma omp atomic
		#endif
		felec += myelec;
	}

	double F = dV * (0.5 * kappa * fgrad + fchem + felec); // Equation 1

	#ifdef MPI_VERSION
	double myF(F);
	MPI::COMM_WORLD.Allreduce(&myF, &F, 1, MPI_DOUBLE, MPI_SUM);
	#endif

	return F;
}

// Discrete Laplacian operator missing the central value, for implicit source terms
template<int dim, typename T>
double fringe_laplacian(const MMSP::grid<dim,MMSP::vector<T> >& GRID, const MMSP::vector<int>& x, const int field)
{
	if (dim > 2) {
		std::cerr << "ERROR: pointerLaplacian is only available for 1- and 2-D fields." << std::endl;
		MMSP::Abort(-1);
	}

	double laplacian = 0.0;

	const double wx = 1.0 / (MMSP::dx(GRID, 0) * MMSP::dx(GRID, 0));
	const int deltax = (dim == 1) ? 1 : 2 * MMSP::ghosts(GRID) + MMSP::y1(GRID) - MMSP::y0(GRID);
	const int deltay = 1;

	const MMSP::vector<T>* const c = &(GRID(x));
	const MMSP::vector<T>* const xl = (MMSP::b0(GRID,0)==MMSP::Neumann && x[0]==MMSP::x0(GRID)  ) ? c : c - deltax;
	const MMSP::vector<T>* const xh = (MMSP::b1(GRID,0)==MMSP::Neumann && x[0]==MMSP::x1(GRID)-1) ? c : c + deltax;

	// No-flux on composition and chemical potential
	if (field != pid) {
		laplacian += wx * ((*xh)[field] + (*xl)[field]);
	} else {
		// External field on electrostatic potential, Equation 13
		if (MMSP::b0(GRID,0)==MMSP::Neumann && x[0]==MMSP::g0(GRID,0)) {
			const double gradPhi = pA * MMSP::dx(GRID,1) * x[1] + pB;
			laplacian += wx * (*xh)[field] - gradPhi / MMSP::dx(GRID,0);
		} else if (MMSP::b1(GRID,0)==MMSP::Neumann && x[0]==MMSP::g1(GRID,0)-1) {
			const double gradPhi = -pA * MMSP::dx(GRID,1) * x[1] - pB;
			laplacian += gradPhi / MMSP::dx(GRID,0) - wx * (*xl)[field];
		} else {
			laplacian += wx * ((*xh)[field] + (*xl)[field]);
		}
	}

	if (dim == 2) {
		const double wy = 1.0 / (MMSP::dx(GRID, 1) * MMSP::dx(GRID, 1));
		const MMSP::vector<T>* const yl = (MMSP::b0(GRID,1)==MMSP::Neumann && x[1]==MMSP::y0(GRID)  ) ? c : c - deltay;
		const MMSP::vector<T>* const yh = (MMSP::b1(GRID,1)==MMSP::Neumann && x[1]==MMSP::y1(GRID)-1) ? c : c + deltay;

		// No-flux on composition and chemical potential
		if (field == cid || field == uid) {
			laplacian += wy * ((*yh)[field] + (*yl)[field]);
		} else {
			// External field on electrostatic potential, Eqn. 13
			if (MMSP::b0(GRID,1)==MMSP::Neumann && x[1]==MMSP::g0(GRID,1)) {
				const double gradPhi = pA * MMSP::dx(GRID,0) * x[0] + pC;
				laplacian += wy * (*yh)[field] - gradPhi / MMSP::dx(GRID,1);
			} else if (MMSP::b1(GRID,1)==MMSP::Neumann && x[1]==MMSP::g1(GRID,1)-1) {
				const double gradPhi = -pA * MMSP::dx(GRID,0) * x[0] - pC;
				laplacian += gradPhi / MMSP::dx(GRID,1) - wy * (*yl)[field];
			} else {
				laplacian += wy * ((*yh)[field] + (*yl)[field]);
			}
		}
	}

	return laplacian;
}

template <int dim, typename T>
MMSP::vector<T> pointerLaplacian(const MMSP::grid<dim,MMSP::vector<T> >& GRID, const MMSP::vector<int>& x)
{
	if (dim > 2) {
		std::cerr << "ERROR: pointerLaplacian is only available for 1- and 2-D fields." << std::endl;
		MMSP::Abort(-1);
	}

	const int N = MMSP::fields(GRID);
	MMSP::vector<T> laplacian(N, 0.0);

	const double wx = 1.0 / (MMSP::dx(GRID, 0) * MMSP::dx(GRID, 0));
	const int deltax = (dim == 1) ? 1 : 2 * MMSP::ghosts(GRID) + MMSP::y1(GRID) - MMSP::y0(GRID);
	const int deltay = 1;

	const MMSP::vector<T>* const c = &(GRID(x));
	const MMSP::vector<T>* const xl = (MMSP::b0(GRID,0)==MMSP::Neumann && x[0]==MMSP::x0(GRID)  ) ? c : c - deltax;
	const MMSP::vector<T>* const xh = (MMSP::b1(GRID,0)==MMSP::Neumann && x[0]==MMSP::x1(GRID)-1) ? c : c + deltax;

	// No-flux on composition and chemical potential
	laplacian[cid] += wx * ((*xh)[cid] - 2. * (*c)[cid] + (*xl)[cid]);
	laplacian[uid] += wx * ((*xh)[uid] - 2. * (*c)[uid] + (*xl)[uid]);

	// External field on electrostatic potential, Eqn. 13
	if (MMSP::b0(GRID,0)==MMSP::Neumann && x[0]==MMSP::g0(GRID,0)) {
		const double gradPhi = pA * MMSP::dx(GRID,1) * x[1] + pB;
		laplacian[pid] += wx * ((*xh)[pid] - (*c)[pid]) - gradPhi / MMSP::dx(GRID,0);
	} else if (MMSP::b1(GRID,0)==MMSP::Neumann && x[0]==MMSP::g1(GRID,0)-1) {
		const double gradPhi = -pA * MMSP::dx(GRID,1) * x[1] - pB;
		laplacian[pid] += gradPhi / MMSP::dx(GRID,0) - wx * ((*c)[pid] - (*xl)[pid]);
	} else {
		laplacian[pid] += wx * ((*xh)[pid] - 2. * (*c)[pid] + (*xl)[pid]);
	}

	if (dim == 2) {
		const double wy = 1.0 / (MMSP::dx(GRID, 1) * MMSP::dx(GRID, 1));
		const MMSP::vector<T>* const yl = (MMSP::b0(GRID,1)==MMSP::Neumann && x[1]==MMSP::y0(GRID)  ) ? c : c - deltay;
		const MMSP::vector<T>* const yh = (MMSP::b1(GRID,1)==MMSP::Neumann && x[1]==MMSP::y1(GRID)-1) ? c : c + deltay;

		// No-flux on composition and chemical potential
		laplacian[cid] += wy * ((*yh)[cid] - 2. * (*c)[cid] + (*yl)[cid]);
		laplacian[uid] += wy * ((*yh)[uid] - 2. * (*c)[uid] + (*yl)[uid]);

		// External field on electrostatic potential, Eqn. 13
		if (MMSP::b0(GRID,1)==MMSP::Neumann && x[1]==MMSP::g0(GRID,1)) {
			const double gradPhi = pA * MMSP::dx(GRID,0) * x[0] + pC;
			laplacian[pid] += wy * ((*yh)[pid] - (*c)[pid]) - gradPhi / MMSP::dx(GRID,1);
		} else if (MMSP::b1(GRID,1)==MMSP::Neumann && x[1]==MMSP::g1(GRID,1)-1) {
			const double gradPhi = -pA * MMSP::dx(GRID,0) * x[0] - pC;
			laplacian[pid] += gradPhi / MMSP::dx(GRID,1) - wy * ((*c)[pid] - (*yl)[pid]);
		} else {
			laplacian[pid] += wy * ((*yh)[pid] - 2. * (*c)[pid] + (*yl)[pid]);
		}
	}

	return laplacian;
}

template<int dim,typename T>
void RedBlackGaussSeidel(const grid<dim,vector<T> >& oldGrid, const T& C0, grid<dim,vector<T> >& newGrid)
{
	int rank=0;
	#ifdef MPI_VERSION
	rank = MPI::COMM_WORLD.Get_rank();
	#endif

	#ifdef DEBUG
	std::ofstream of;
	if (rank == 0)
		of.open("iter.log", std::ofstream::out | std::ofstream::app); // new results will be appended
	#endif

	double gridSize = static_cast<double>(nodes(oldGrid));
	#ifdef MPI_VERSION
	double localGridSize = gridSize;
	MPI::COMM_WORLD.Allreduce(&localGridSize, &gridSize, 1, MPI_DOUBLE, MPI_SUM);
	#endif

	double dV = 1.0;
	double lapWeight = 0.0;
	for (int d=0; d<dim; d++) {
		dV *= dx(oldGrid,d);
		lapWeight += 2.0 / std::pow(dx(oldGrid, d), 2.0);
	}

	double residual = 2.0;
	unsigned int iter = 0;

	while (iter < max_iter && residual > tolerance) {
		/*  ==== RED-BLACK GAUSS SEIDEL ====
		    Iterate over a checkerboard, updating first red then black tiles.
		    If the sum of indices is even, then the tile is Red; else, Black.

			This method solves the linear system of equations,
		    [ a11 a12  0  ][ x1 ]   [ b1 ]
		    [ a21  1  a23 ][ x2 ] = [ b2 ]
			[ a31  0  a33 ][ x3 ]   [ b3 ]
		*/

		for (int color = 1; color > -1; color--) {
			// If color==1, skip BLACK tiles, which have Σx[d] odd
			// If color==0, skip RED tiles, which have Σx[d] even
			#ifdef _OPENMP
			#pragma omp parallel for
			#endif
			for (int n=0; n<nodes(oldGrid); n++) {
				// Within these iterations, "x_n" --> "xOld" and "x_{n+1}" --> "xGuess".
				vector<int> x = position(oldGrid,n);
				const double xx = dx(oldGrid,0) * x[0];
				const double yy = dx(oldGrid,1) * x[1];
				double myLapWeight = lapWeight;
				for (int d=0; d<dim; d++) {
					if (MMSP::b0(oldGrid,d)==MMSP::Neumann && x[d]==MMSP::g0(oldGrid,d)) {
						myLapWeight -= 1.0 / std::pow(dx(oldGrid, d), 2.0);
					} else if (MMSP::b1(oldGrid,d)==MMSP::Neumann && x[d]==MMSP::g1(oldGrid,d)-1) {
						myLapWeight -= 1.0 / std::pow(dx(oldGrid, d), 2.0);
					}
				}

				// Determine tile color
				int x_sum=0;
				for (int d=0; d<dim; d++)
					x_sum += x[d];
				if (x_sum % 2 == color)
					continue;

				const T cOld = oldGrid(n)[cid];

				const T cGuess = newGrid(n)[cid]; // value from last "guess" iteration
				const T uGuess = newGrid(n)[uid];
				const T pGuess = newGrid(n)[pid];

				const vector<T> gradC = gradient(newGrid, x, cid);
				const vector<T> gradU = gradient(newGrid, x, uid);
				const double dotgradCU = gradC * gradU;
				const double M = M0 / (1.0 + cGuess * cGuess);
				const double dMdc = -2. * M0 / std::pow(1.0 + cGuess * cGuess, 2.);

				// A is defined by the last guess, stored in newGrid(n).
				// It is a 3x3 matrix.
				const double a11 = 1. - dMdc * dt * dotgradCU;
				const double a12 = lapWeight * dt * M;
				const double a21 = -kappa * lapWeight - dfcontractivedc(cGuess, 1.0);
				const double a22 = 1.0;
				const double a23 = -k;
				const double a31 = k / epsilon;
				const double a33 = -myLapWeight;

				// B is defined by the last value, stored in oldGrid(n), and the
				// last guess, stored in newGrid(n). It is a 3x1 column.
				const double flapC = fringe_laplacian(newGrid, x, cid);
				const double flapU = fringe_laplacian(newGrid, x, uid);
				const double flapP = fringe_laplacian(newGrid, x, pid);

				const double b1 = cOld + dt * M * flapU;
				const double b2 = dfexpansivedc(cOld) - kappa * flapC + k * pExt(xx, yy);
				const double b3 = k * C0 / epsilon - flapP;

				// Solve the iteration system AX=B using Cramer's rule
				const double detA  = a33 * (a11 * a22 - a12 * a21) + a12 * a23 * a31;
				const double detA1 = a33 * (b1  * a22 - a12 * b2 ) + a12 * a23 * b3;
				const double detA2 = a33 * (a11 * b2  - b1  * a21) + a23 * (b1 * a31 - a11 * b3);
				const double detA3 = a22 * (a11 * b3  - b1  * a31) + a12 * (b2 * a31 - a21 * b3);

				const T cNew = detA1 / detA;
				const T uNew = detA2 / detA;
				const T pNew = detA3 / detA;

				// (Don't) Apply relaxation
				newGrid(n)[cid] = omega * cNew + (1.0 - omega) * cGuess;
				newGrid(n)[uid] = omega * uNew + (1.0 - omega) * uGuess;
				newGrid(n)[pid] = omega * pNew + (1.0 - omega) * pGuess;

			}
			ghostswap(newGrid);
		}

		iter++;

		/*  ==== RESIDUAL ====
		    The residual is computed from the original matrix form, Ax=b:
		    any Old term goes into B, while any New term goes in AX. Note that
		    this is not the iteration matrix, it is the original system of equations.
		*/

		if (iter % residual_step == 0) {
			double normB = 0.0;
			residual = 0.0;

			#ifdef _OPENMP
			#pragma omp parallel for
			#endif
			for (int n=0; n<nodes(newGrid); n++) {
				vector<int> x = position(newGrid,n);
				const double xx = dx(newGrid,0) * x[0];
				const double yy = dx(newGrid,1) * x[1];
				const vector<T> lap = pointerLaplacian(newGrid, x);

				const T cOld = oldGrid(n)[cid];

				const T cNew = newGrid(n)[cid];
				const T uNew = newGrid(n)[uid];
				const T pNew = newGrid(n)[pid];

				const vector<T> gradC = gradient(newGrid, x, cid);
				const vector<T> gradU = gradient(newGrid, x, uid);
				const double dotgradCU = gradC * gradU;

				const double M = M0 / std::pow(1.0 + cNew * cNew, 2.);
				const double dMdc = M0 / (1.0 + cNew * cNew);

				// Plug iteration results into original system of equations
				const double Ax1 = (1. - dMdc * dt * dotgradCU) * cNew
				                 - (M * dt) * lap[uid];
				const double Ax2 = uNew + kappa * lap[cid] - dfcontractivedc(cNew, cNew) - k * pNew;
				const double Ax3 = lap[pid] + k * cNew / epsilon;

				const double b1 = cOld;
				const double b2 = dfexpansivedc(cOld) + k * pExt(xx, yy);
				const double b3 = k * C0 / epsilon;

				// Compute the Error from parts of the solution
				const double r1 = b1 - Ax1;
				const double r2 = b2 - Ax2;
				const double r3 = b3 - Ax3;

				const double error  = r1*r1 + r2*r2 + r3*r3;
				const double source = b1*b1 + b2*b2 + b3*b3;

				#ifdef _OPENMP
				#pragma omp atomic
				#endif
				residual += error;

				#ifdef _OPENMP
				#pragma omp atomic
				#endif
				normB += source;
			}

			#ifdef MPI_VERSION
			double localResidual = residual;
			MPI::COMM_WORLD.Allreduce(&localResidual, &residual, 1, MPI_DOUBLE, MPI_SUM);
			double localNormB = normB;
			MPI::COMM_WORLD.Allreduce(&localNormB, &normB, 1, MPI_DOUBLE, MPI_SUM);
			#endif

			residual = (std::fabs(normB) > tolerance) ? sqrt(residual/normB)/(3.0*gridSize) : 0.0;

			if (iter % residual_step == 0 || residual < tolerance) {
				double F = Helmholtz(newGrid, C0);
				#ifdef MPI_VERSION
				double localF(F);
				MPI::COMM_WORLD.Allreduce(&localF, &F, 1, MPI_DOUBLE, MPI_SUM);
				#endif
				#ifdef DEBUG
				if (rank == 0)
					of << iter << '\t' << residual << '\t' << F << std::endl;
				#endif
			}
		}
	}

	#ifdef DEBUG
	if (rank == 0)
		of.close();
	#endif

	#ifdef MPI_VERSION
	unsigned int myit(iter);
	MPI::COMM_WORLD.Allreduce(&myit, &iter, 1, MPI_UNSIGNED, MPI_MAX);
	#endif

	if (iter >= max_iter) {
		if (rank==0)
			std::cerr << "Solver stagnated. Aborting." << std::endl;
		MMSP::Abort(-1);
	}
}

template<int dim,typename T>
void PoissonSolver(grid<dim,vector<T> >& GRID, const double C0)
{
	// Iterative Poisson solver after http://yyy.rsmas.miami.edu/users/miskandarani/Courses/MSC321/Projects/prjpoisson.pdf

	#ifdef DEBUG
	int rank=0;
	#ifdef MPI_VERSION
	rank = MPI::COMM_WORLD.Get_rank();
	#endif
	#endif

	MMSP::grid<dim,double> poisGrid(GRID,0);
	for (int d=0; d<dim; d++)
		dx(poisGrid, d) = deltaX;
	for (int n=0; n<nodes(poisGrid); n++) {
		// Set initial field to external field
		vector<int> x = position(poisGrid, n);
		poisGrid(n) = GRID(n)[pid];
	}

	#ifdef DEBUG
	std::ofstream of;
	if (rank == 0)
		of.open("iter.log", std::ofstream::out|std::ios_base::app);
	#endif

	double res = 1.0;
	unsigned int iter = 0;
	while (res > tolerance) {
		#ifdef _OPENMP
		#pragma omp parallel for
		#endif
		for (int n=0; n<nodes(GRID); n++) {
			vector<int> x = position(GRID, n);
			const int deltax = (dim == 1) ? 1 : 2 * MMSP::ghosts(GRID) + MMSP::y1(GRID) - MMSP::y0(GRID);
			int deltay = 1;

			const double* const c = &(poisGrid(x));
			const double* const xl = (MMSP::b0(GRID,0)==MMSP::Neumann && x[0]==MMSP::x0(GRID)  ) ? c : c - deltax;
			const double* const xh = (MMSP::b1(GRID,0)==MMSP::Neumann && x[0]==MMSP::x1(GRID)-1) ? c : c + deltax;

			// initialize right-hand side
			double rhs = deltaX*deltaX * k * (GRID(n)[cid] - C0) / epsilon;
			int denom = 4;

			if (MMSP::b0(GRID,0)==MMSP::Neumann && x[0]==MMSP::g0(GRID,0)) {
				// Left boundary
				const double gradPhi = -pA * dx(GRID,1) * x[1] - pB;
				rhs += (*xh) - deltaX * gradPhi;
				denom--;
			} else if (MMSP::b1(GRID,0)==MMSP::Neumann && x[0]==MMSP::g1(GRID,0)-1) {
				// Right boundary
				const double gradPhi = pA * dx(GRID,1) * x[1] + pB;
				rhs += (*xl) - deltaX * gradPhi;
				denom--;
			} else {
				rhs += (*xh) + (*xl);
			}

			if (dim == 2) {
				const double* const yl = (MMSP::b0(GRID,1)==MMSP::Neumann && x[1]==MMSP::y0(GRID)  ) ? c : c - deltay;
				const double* const yh = (MMSP::b1(GRID,1)==MMSP::Neumann && x[1]==MMSP::y1(GRID)-1) ? c : c + deltay;

				if (MMSP::b0(GRID,1)==MMSP::Neumann && x[1]==MMSP::g0(GRID,1)) {
					// Bottom boundary
					const double gradPhi = -pA * dx(GRID,0) * x[0] - pC;
					rhs += (*yh) - deltaX * gradPhi;
					denom--;
				} else if (MMSP::b1(GRID,1)==MMSP::Neumann && x[1]==MMSP::g1(GRID,1)-1) {
					// Top boundary
					const double gradPhi = pA * dx(GRID,0) * x[0] + pC;
					rhs += (*yl) - deltaX * gradPhi;
					denom--;
				} else {
					rhs += (*yh) + (*yl);
				}
			}

			poisGrid(n) = rhs / denom;
		}

		iter++;

		ghostswap(poisGrid);

		if (iter < 10 || iter % 10 == 0) {
			// residual
			res = 0.0;
			double norm = 0.0;

			#ifdef _OPENMP
			#pragma omp parallel for
			#endif
			for (int n=0; n<nodes(GRID); n++) {
				vector<int> x = position(GRID, n);
				const int deltax = (dim == 1) ? 1 : 2 * MMSP::ghosts(GRID) + MMSP::y1(GRID) - MMSP::y0(GRID);
				const int deltay = 1;

				const double* const c = &(poisGrid(x));
				const double* const xl = (MMSP::b0(poisGrid,0)==MMSP::Neumann && x[0]==MMSP::x0(poisGrid)  ) ? c : c - deltax;
				const double* const xh = (MMSP::b1(poisGrid,0)==MMSP::Neumann && x[0]==MMSP::x1(poisGrid)-1) ? c : c + deltax;
				const double wx = 1.0 / (deltaX * deltaX);

				double lap = 0.;

				if (MMSP::b0(poisGrid,0)==MMSP::Neumann && x[0]==MMSP::g0(poisGrid,0)) {
					// Left boundary
					const double gradPhi = -pA * dx(GRID,1) * x[1] - pB;
					lap += wx * (*xh - *c) - gradPhi / deltaX;
				} else if (MMSP::b1(poisGrid,0)==MMSP::Neumann && x[0]==MMSP::g1(poisGrid,0)-1) {
					// Right boundary
					const double gradPhi = pA * dx(GRID,1) * x[1] + pB;
					lap += -gradPhi / deltaX - wx * (*c - *xl);
				} else {
					lap += wx * (*xh + *xl - 2. * *c);
				}

				if (dim == 2) {
					const double* const yl = (MMSP::b0(poisGrid,1)==MMSP::Neumann && x[1]==MMSP::y0(poisGrid)  ) ? c : c - deltay;
					const double* const yh = (MMSP::b1(poisGrid,1)==MMSP::Neumann && x[1]==MMSP::y1(poisGrid)-1) ? c : c + deltay;
					const double wy = 1.0 / (deltaX * deltaX);

					if (MMSP::b0(poisGrid,1)==MMSP::Neumann && x[1]==MMSP::g0(poisGrid,1)) {
						// Bottom boundary
						const double gradPhi = -pA * dx(GRID,0) * x[0] - pC;
						lap += wy * (*yh - *c) - gradPhi / deltaX;
					} else if (MMSP::b1(poisGrid,1)==MMSP::Neumann && x[1]==MMSP::g1(poisGrid,1)-1) {
						// Top boundary
						const double gradPhi = pA * dx(GRID,0) * x[0] + pC;
						lap += -gradPhi / deltaX - wy * (*c - *yl);
					} else {
						lap += wy * (*yh + *yl - 2. * *c);
					}
				}
				double rhs =  -k * (GRID(n)[cid] - C0) / epsilon;

				#ifdef _OPENMP
				#pragma omp atomic
				#endif
				res += (rhs - lap) * (rhs - lap);

				#ifdef _OPENMP
				#pragma omp atomic
				#endif
				norm += rhs * rhs;
			}

			res = std::sqrt(res / norm) / nodes(GRID);

			#ifdef DEBUG
			if (iter < 10 || iter % residual_step == 0 || res < tolerance) {
				const double F = Helmholtz(GRID, C0);

				#ifdef MPI_VERSION
				double localF(F);
				MPI::COMM_WORLD.Allreduce(&localF, &F, 1, MPI_DOUBLE, MPI_SUM);
				#endif
				of << iter << '\t' << res << '\t' << F << std::endl;
			}
			#endif
		}
	}

	#ifdef _OPENMP
	#pragma omp parallel for
	#endif
	for (int n=0; n<nodes(GRID); n++)
		GRID(n)[pid] = poisGrid(n);

	#ifdef DEBUG
	if (rank == 0)
		of.close();
	#endif
}

void generate(int dim, const char* filename)
{
	/*
	  Field 0 is [c] composition.
	  Field 1 is [u] chemical potential.
	  Field 2 is [p] electrostatic potential.
	*/

	int rank=0;
	#ifdef MPI_VERSION
	rank = MPI::COMM_WORLD.Get_rank();
	#endif

	if (dim != 2 && rank == 0) {
		std::cerr << "ERROR: CHiMaD problems are 2-D, only!" << std::endl;
		std::exit(-1);
	}

	#ifdef DEBUG
	std::ofstream of;
	if (rank == 0) {
		of.open("iter.log", std::ofstream::out);
		of.close();
	}
	#endif

	if (dim==2) {
		const int L = 100 / deltaX;
		GRID2D initGrid(3, 0,L, 0,L);
		double gridSize = 1.0;
		for (int d=0; d<dim; d++) {
			// Set grid resolution
			dx(initGrid, d) = deltaX;
			gridSize *= double(g1(initGrid, d) - g0(initGrid, d));

			// Set Neumann (zero-flux) boundary conditions
			if (x0(initGrid, d) == g0(initGrid, d))
				b0(initGrid, d) = Neumann;
			if (x1(initGrid, d) == g1(initGrid, d))
				b1(initGrid, d) = Neumann;
		}

		if (rank == 0)
			std::cout << "Timestep is " << dt << ". CFL is " << CFL
					  << ". Run " << 1.0 / dt << " per unit time."
					  << std::endl;

		#ifdef _OPENMP
		#pragma omp parallel for
		#endif
		for (int n=0; n<nodes(initGrid); n++) {
			vector<int> x = position(initGrid, n);
			// composition field
			initGrid(n)[cid] = cheminit(dx(initGrid,0) * x[0], dx(initGrid,1) * x[1]);
			initGrid(n)[pid] = pExt(deltaX * x[0], deltaX * x[1]);
		}

		ghostswap(initGrid);

		const double c0 = system_composition(initGrid);
		std::cout << "System composition is " << c0 << std::endl;

		#ifdef _OPENMP
		#pragma omp parallel for
		#endif
		for (int n=0; n<nodes(initGrid); n++) {
			vector<int> x = position(initGrid, n);
			const double xx = dx(initGrid,0) * x[0];
			const double yy = dx(initGrid,1) * x[1];
			const double c = initGrid(n)[cid];
			const double lapC = laplacian(initGrid, x, cid);
			initGrid(n)[uid] = 2. * w * (c-Ca) * (Cb-c) * (Ca+Cb-2.*c) - kappa*lapC + k*(initGrid(n)[pid] + pExt(xx, yy));
		}

		PoissonSolver(initGrid, c0);

		output(initGrid,filename);
	}
}

template <int dim, typename T>
void update(grid<dim,vector<T> >& oldGrid, int steps)
{
	/*	Grid contains three fields:
		0: composition
		1: chemical potential
		2: electrostatic potential  */

	int rank=0;
	#ifdef MPI_VERSION
	rank = MPI::COMM_WORLD.Get_rank();
	#endif

	#ifndef DEBUG
	static double elapsed = 0.0;
	#endif

	ghostswap(oldGrid);

	grid<dim,vector<T> > newGrid(oldGrid);
	newGrid.copy(oldGrid);

	// Make sure the grid spacing is correct
	double gridSize = 1.0;
	for (int d=0; d<dim; d++) {
		dx(oldGrid,d) = deltaX;
		dx(newGrid,d) = deltaX;
		gridSize *= double(g1(oldGrid, d) - g0(oldGrid, d));

		// Set Neumann (zero-flux) boundary conditions
		if (x0(oldGrid, d) == g0(oldGrid, d)) {
			b0(oldGrid, d) = Neumann;
			b0(newGrid, d) = Neumann;
		}
		if (x1(oldGrid, d) == g1(oldGrid, d)) {
			b1(oldGrid, d) = Neumann;
			b1(newGrid, d) = Neumann;
		}
	}

	#ifndef DEBUG
	std::ofstream of;
	if (rank == 0)
		of.open("energy_dx02.tsv", std::ofstream::out | std::ofstream::app); // new results will be appended
	#endif

	for (int step=0; step<steps; step++) {
		if (rank==0)
			print_progress(step, steps);

		const double c0 = system_composition(oldGrid);

		// ghostswap(newGrid);
		// RedBlackGaussSeidel(oldGrid, c0, newGrid);

		for (int n=0; n < nodes(oldGrid); n++) {
			vector<int> x = position(oldGrid, n);
			const T& cOld = oldGrid(n)[cid];
			const T& pOld = oldGrid(n)[pid];
			const vector<T> gradC = gradient(newGrid, x, cid);
			const vector<T> gradU = gradient(newGrid, x, uid);
			const double dotgradCU = gradC * gradU;
			const double M = M0 / (1.0 + cOld * cOld);
			const double dMdc = -2. * M0 / std::pow(1.0 + cOld * cOld, 2.);
			const vector<T> lap = pointerLaplacian(oldGrid, x);

			newGrid(n)[cid] = cOld + dt * (dMdc * dotgradCU + M * lap[uid]);
			newGrid(n)[uid] = 2. * w * (cOld - Ca) * (Cb - cOld) * (Ca + Cb - 2. * cOld)
				            - kappa * lap[cid]
				            + k * (pOld + pExt(dx(oldGrid,0) * x[0], dx(oldGrid,1) * x[1]));
		}

		swap(oldGrid, newGrid);
		ghostswap(oldGrid);
		PoissonSolver(oldGrid, c0);

		#ifndef DEBUG
		elapsed += dt;
		const double F = Helmholtz(oldGrid, c0);
		if (rank == 0)
			of << elapsed << '\t' << c0 << '\t' << F << std::endl;
		#endif
	}

	#ifndef DEBUG
	if (rank == 0)
		of.close();
	#endif

}

} // MMSP

#endif

#include"MMSP.main.hpp"
