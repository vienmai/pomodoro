#include <iostream>
#include "algebra.hpp"
#include "matrix.hpp"

using namespace matrix;

int main(int argc, char *argv[]) {

	int n=3;
	dmatrix<double> A(n,n);
	std::ifstream infile("A.txt", std::ifstream::binary);
	A.load(infile);

	// dmatrix<double> A(n, n, {1, 0, 3., 0, 1, 0, 2, 0, 1});
	std::vector<double> x{1, -2, 1};
	std::vector<double> y{0, 0, 0};

	int nrows = A.nrows();
	int ncols = A.ncols();

	printf("nrows = %d,  ncols = %d\n", nrows, ncols);
	for (int i = 0; i < nrows; i++)
	{
		for (int j = 0; j < ncols; j++)
			std::cout << A(i, j) << " ";
		std::cout << std::endl;
	}

	A.mult_add('n', 1., &x[0], 1., &y[0]);

	std::cout << "y = Ax " << std::endl;
	for (int i = 0; i < nrows; i++)
		std::cout << y[i] << " ";
	std::cout << std::endl;

	std::ofstream outfile("A.txt", std::ofstream::binary);
	A.save(outfile);

	return 0;
}
