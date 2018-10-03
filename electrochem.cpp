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
const double deltaX = 0.2;
const double dt = 0.05;
const double CFL = (24.0 * M0 * kappa) / (2.0 * std::pow(deltaX, 4));

template <int dim,typename T>
double Helmholtz(const grid<dim,vector<T> >& GRID, const T& C0)
{
	double dV = 1.0;
	for (int d=0; d<dim; d++)
		dV *= dx(GRID, d);

	double fchem = 0.0;
	double felec = 0.0;
	double fgrad = 0.0;

	for (int n=0; n<nodes(GRID); n++) {
		vector<int> x = position(GRID, n);
		vector<double> gradc = gradient(GRID, x, cid);

		fchem += chemenergy(GRID(n)[cid]);
		felec += elecenergy(GRID(n)[cid], C0, GRID(n)[pid]);
		fgrad += gradc * gradc;
	}

	double F = dV * (fchem + felec + 0.5 * kappa * fgrad); // Equation 5

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
	const MMSP::vector<T>* const l = (MMSP::b0(GRID,0)==MMSP::Neumann && x[0]==MMSP::x0(GRID)  ) ? c : c - deltax;
	const MMSP::vector<T>* const h = (MMSP::b1(GRID,0)==MMSP::Neumann && x[0]==MMSP::x1(GRID)-1) ? c : c + deltax;

	// No-flux on composition and chemical potential
	if (field == cid || field == uid) {
		laplacian += wx * ((*h)[field] + (*l)[field]);
	} else {
		// External field on electrostatic potential, Eqn. 13
		if (MMSP::b0(GRID,0)==MMSP::Neumann && x[0]==MMSP::x0(GRID)) {
			laplacian += wx * (*h)[field]
			          +  0.0002 * MMSP::dx(GRID,1) * x[1] - 0.01
		              +  0.0002 * MMSP::dx(GRID,0) * x[0] + 0.02;
		} else if (MMSP::b1(GRID,0)==MMSP::Neumann && x[0]==MMSP::x1(GRID)-1) {
			laplacian += 0.0002 * MMSP::dx(GRID,1) * x[1] - 0.01
			          +  0.0002 * MMSP::dx(GRID,0) * x[0] + 0.02
			          +  wx * (*l)[field];
		} else {
			laplacian += wx * ((*h)[field] + (*l)[field]);
	}
	}

	if (dim == 2) {
		const double wy = 1.0 / (MMSP::dx(GRID, 1) * MMSP::dx(GRID, 1));
		const MMSP::vector<T>* const cl = (MMSP::b0(GRID,1)==MMSP::Neumann && x[1]==MMSP::y0(GRID)  ) ? c : c - deltay;
		const MMSP::vector<T>* const ch = (MMSP::b1(GRID,1)==MMSP::Neumann && x[1]==MMSP::y1(GRID)-1) ? c : c + deltay;

		// No-flux on composition and chemical potential
		if (field == cid || field == uid) {
			laplacian += wy * ((*ch)[field] + (*cl)[field]);
		} else {
			// External field on electrostatic potential, Eqn. 13
			if (MMSP::b0(GRID,0)==MMSP::Neumann && x[0]==MMSP::x0(GRID)) {
				laplacian += wy * (*ch)[field]
				          +  0.0002 * MMSP::dx(GRID,1) * x[1] - 0.01
				          +  0.0002 * MMSP::dx(GRID,0) * x[0] + 0.02;
			} else if (MMSP::b1(GRID,0)==MMSP::Neumann && x[0]==MMSP::x1(GRID)-1) {
				laplacian += 0.0002 * MMSP::dx(GRID,1) * x[1] - 0.01
				          +  0.0002 * MMSP::dx(GRID,0) * x[0] + 0.02
				          +  wy * (*l)[field];
			} else {
				laplacian += wy * ((*h)[field] + (*l)[field]);
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
	const MMSP::vector<T>* const l = (MMSP::b0(GRID,0)==MMSP::Neumann && x[0]==MMSP::x0(GRID)  ) ? c : c - deltax;
	const MMSP::vector<T>* const h = (MMSP::b1(GRID,0)==MMSP::Neumann && x[0]==MMSP::x1(GRID)-1) ? c : c + deltax;

	// No-flux on composition and chemical potential
	laplacian[cid] += wx * ((*h)[cid] - 2. * (*c)[cid] + (*l)[cid]);
	laplacian[uid] += wx * ((*h)[uid] - 2. * (*c)[uid] + (*l)[uid]);

	// External field on electrostatic potential, Eqn. 13
	if (MMSP::b0(GRID,0)==MMSP::Neumann && x[0]==MMSP::x0(GRID)) {
		laplacian[pid] += wx * ((*h)[pid] - (*c)[pid])
			           +  0.0002 * MMSP::dx(GRID,1)*x[1] - 0.01
		               +  0.0002 * MMSP::dx(GRID,0)*x[0] + 0.02;
	} else if (MMSP::b1(GRID,0)==MMSP::Neumann && x[0]==MMSP::x1(GRID)-1) {
		laplacian[pid] += 0.0002*MMSP::dx(GRID,1)*x[1] - 0.01
			           +  0.0002*MMSP::dx(GRID,0)*x[0] + 0.02
			           -  wx * ((*c)[pid] - (*l)[pid]);
	} else {
		laplacian[pid] += wx * ((*h)[pid] - 2. * (*c)[pid] + (*l)[pid]);
	}

	if (dim == 2) {
		const double wy = 1.0 / (MMSP::dx(GRID, 1) * MMSP::dx(GRID, 1));
		const MMSP::vector<T>* const cl = (MMSP::b0(GRID,1)==MMSP::Neumann && x[1]==MMSP::y0(GRID)  ) ? c : c - deltay;
		const MMSP::vector<T>* const ch = (MMSP::b1(GRID,1)==MMSP::Neumann && x[1]==MMSP::y1(GRID)-1) ? c : c + deltay;

		// No-flux on composition and chemical potential
		laplacian[cid] += wy * ((*ch)[cid] - 2. * (*c)[cid] + (*cl)[cid]);
		laplacian[uid] += wy * ((*ch)[uid] - 2. * (*c)[uid] + (*cl)[uid]);

		// External field on electrostatic potential, Eqn. 13
		if (MMSP::b0(GRID,0)==MMSP::Neumann && x[0]==MMSP::x0(GRID)) {
			laplacian[pid] += wy * ((*ch)[pid] - (*c)[pid])
				           +  0.0002 * MMSP::dx(GRID,1) * x[1] - 0.01
				           +  0.0002 * MMSP::dx(GRID,0) * x[0] + 0.02;
		} else if (MMSP::b1(GRID,0)==MMSP::Neumann && x[0]==MMSP::x1(GRID)-1) {
			laplacian[pid] += 0.0002 * MMSP::dx(GRID,1) * x[1] - 0.01
				           +  0.0002 * MMSP::dx(GRID,0) * x[0] + 0.02
				           -  wy * ((*c)[pid] - (*l)[pid]);
		} else {
			laplacian[pid] += wy * ((*h)[pid] - 2. * (*c)[pid] + (*l)[pid]);
		}
	}

	return laplacian;
}

template<int dim,typename T>
unsigned int RedBlackGaussSeidel(const grid<dim,vector<T> >& oldGrid, const T& C0, grid<dim,vector<T> >& newGrid)
{
	#ifdef DEBUG
	int rank=0;
	#ifdef MPI_VERSION
	rank = MPI::COMM_WORLD.Get_rank();
	#endif

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
		    [ a11  a12 0  ][ x1 ]   [ b1 ]
		    [ a21  1   0  ][ x2 ] = [ b2 ]
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
				double myLapWeight = lapWeight;
				if (MMSP::b0(oldGrid,0)==MMSP::Neumann && x[0]==MMSP::x0(oldGrid)) {
					myLapWeight -= 1.0 / std::pow(dx(oldGrid, 0), 2.0);
				} else if (MMSP::b1(oldGrid,0)==MMSP::Neumann && x[0]==MMSP::x1(oldGrid)-1) {
					myLapWeight -= 1.0 / std::pow(dx(oldGrid, 1), 2.0);
				}

				// Determine tile color
				int x_sum=0;
				for (int d=0; d<dim; d++)
					x_sum += x[d];
				if (x_sum % 2 == color)
					continue;

				const T cOld = oldGrid(n)[cid];
				const T pOld = oldGrid(n)[pid];

				const T cGuess = newGrid(n)[cid]; // value from last "guess" iteration
				const T uGuess = newGrid(n)[uid];
				T pGuess = newGrid(n)[pid];

				const vector<T> gradC = gradient(oldGrid, x, cid);
				const vector<T> gradU = gradient(oldGrid, x, uid);
				const double dotgradCU = gradC * gradU;

				// A is defined by the last guess, stored in newGrid(n).
				// It is a 3x3 matrix.
				const double a11 = 1. + (2. * dotgradCU * M0 * dt) / std::pow(1. + cOld*cOld, 2.);
				const double a12 = myLapWeight * dt * M0 / (1.0 + cOld * cOld);
				const double a21 = -kappa * myLapWeight - dfcontractivedc(cOld, 1.0);
				const double a31 = k / epsilon;
				const double a33 = -myLapWeight;

				// B is defined by the last value, stored in oldGrid(n), and the
				// last guess, stored in newGrid(n). It is a 3x1 column.
				const double flapU = fringe_laplacian(newGrid, x, uid);
				const double flapP = fringe_laplacian(newGrid, x, pid);

				const double b1 = cOld + dt * flapU;
				const double b2 = dfexpansivedc(cOld, C0, pOld) + k * pOld + k * pExt(oldGrid, x);
				const double b3 = k * cOld / epsilon - flapP;

				// Solve the iteration system AX=B using Cramer's rule
				const double detA  = a33 * (a11 - a12 * a21);
				const double detA1 = a33 * (b1 - a12 * b2 );
				const double detA2 = a33 * (b2 - b1  * a21);
				const double detA3 = 1.0 * (a11 - b1 * a31) + a12 * (b2 * a31 - a21 * b3);

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

		#ifdef DEBUG
		if (iter % 10 ==0) {
			if (rank == 0) {
				std::cout << "Iter. " << iter << ", " << std::flush;
			}
		}
		#endif

		/*  ==== RESIDUAL ====
		    The residual is computed from the original matrix form, Ax=b:
		    any Old term goes into B, while any New term goes in AX. Note that
		    this is not the iteration matrix, it is the original system of equations.
		*/

		if (iter < residual_step || iter % residual_step == 0) {
			double normB = 0.0;
			residual = 0.0;

			#ifdef _OPENMP
			#pragma omp parallel for
			#endif
			for (int n=0; n<nodes(newGrid); n++) {
				vector<int> x = position(newGrid,n);
				const vector<T> lap = pointerLaplacian(newGrid, x);

				const T cOld = oldGrid(n)[cid];
				const T pOld = oldGrid(n)[pid];

				const T cNew = newGrid(n)[cid];
				const T uNew = newGrid(n)[uid];

				const vector<T> gradC = gradient(oldGrid, x, cid);
				const vector<T> gradU = gradient(oldGrid, x, uid);
				const double dotgradCU = gradC * gradU;

				const double M = M0 / std::pow(1.0 + cOld*cOld, 2.);

				// Plug iteration results into original system of equations
				const double Ax1 = (1. + 2. * M * dt * dotgradCU) * cNew
					             - (M0 * dt / (1. + cNew*cNew)) * lap[uid];
				const double Ax2 = uNew + kappa * lap[cid] - dfcontractivedc(cNew, cNew);
				const double Ax3 = lap[uid] + k * cNew / epsilon;

				const double b1 = cOld;
				const double b2 = dfexpansivedc(cOld, C0, pOld) + k * pOld + k * pExt(oldGrid, x);
				const double b3 = k * cOld / epsilon;

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

			#ifdef DEBUG
			double F = Helmholtz(oldGrid, C0);

			if (rank == 0)
				of << iter << '\t' << residual << '\t' << F << '\n';
			#endif

			#ifdef DEBUG
			if (iter % 10 ==0) {
				std::cout << " res. " << residual << std::endl;
			}
			#endif
		}
	}

	#ifdef DEBUG
	if (rank == 0)
		of.close();
	#endif

	return iter;
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

	if (dim==2) {
		const int L = 100 / deltaX;
		GRID2D initGrid(3, 0,L, 0,L);
		for (int d=0; d<dim; d++) {
			// Set grid resolution
			dx(initGrid,d) = deltaX;

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
		}

		ghostswap(initGrid);

		double c0 = 0.0;
		for (int n=0; n<nodes(initGrid); n++)
			c0 += initGrid(n)[cid];
		c0 /= nodes(initGrid);
		std::cout << "System composition is " << c0 << std::endl;

		#ifdef _OPENMP
		#pragma omp parallel for
		#endif
		for (int n=0; n<nodes(initGrid); n++) {
			vector<int> x = position(initGrid, n);
			const double c = initGrid(n)[cid];
			const double lapC = laplacian(initGrid, x, cid);
			initGrid(n)[pid] = -k * (initGrid(n)[cid] - c0) / epsilon;
			initGrid(n)[uid] = 2. * w * (c-Ca) * (Cb-c) * (Ca+Cb-2.*c) - kappa*lapC + k*(initGrid(n)[pid] + pExt(initGrid, x));
		}

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
	for (int d=0; d<dim; d++) {
		dx(oldGrid,d) = deltaX;
		dx(newGrid,d) = deltaX;

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

	double c0 = 0.0;
	for (int n=0; n<nodes(oldGrid); n++)
		c0 += oldGrid(n)[cid];
	c0 /= nodes(oldGrid);

	#ifndef DEBUG
	std::ofstream of;
	if (rank == 0)
		of.open("energy_dx02.tsv", std::ofstream::out | std::ofstream::app); // new results will be appended
	#endif

	for (int step=0; step<steps; step++) {
		if (rank==0)
			print_progress(step, steps);

		ghostswap(newGrid);

		unsigned int iter = RedBlackGaussSeidel(oldGrid, c0, newGrid);

		#ifdef MPI_VERSION
		unsigned int myit(iter);
		MPI::COMM_WORLD.Allreduce(&myit, &iter, 1, MPI_UNSIGNED, MPI_MAX);
		#endif

		if (iter >= max_iter) {
			if (rank==0)
				std::cerr << "Solver stagnated on step " << step << ". Aborting." << std::endl;
			MMSP::Abort(-1);
		}

		swap(oldGrid, newGrid);
		ghostswap(oldGrid);

		#ifndef DEBUG
		elapsed += dt;
		const double F = Helmholtz(oldGrid, c0);
		if (rank == 0)
			of << elapsed << '\t' << F << '\n';
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
