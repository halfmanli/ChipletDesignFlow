// Copyright 2017 by Robert Dick.
// All rights reserved.

#include "Epsilon.h"
#include "RStd.h"

namespace rstd {
using namespace std;

/*###########################################################################*/
void Epsilon_test() {
#if 0
	eps_less<double, 1> lt;
	double x = 1.0;
	double y = 1.00000000001;
	cout.precision(20);
	while (1) {
		cout << y << " " << x << endl;
		if (lt(y, x)) {
			cout << "Finally lt" << endl;
			return;
		}
		y -= 0.000000000000001;
	}

#else
	double x = 50000;
	double y = 50000.0000001;

	eps_less<double, 10> lt;
	eps_equal_to<double, 10> eq;
	eps_not_equal_to<double, 10> neq;

	Rassert(! lt(x, y));
	Rassert(eq(x, y));

	double a = 500;
	double b = 501;

	Rassert(lt(a, b));
	Rassert(! eq(a, b));

	Rassert(neq(50.0, 51.0));
	
	double x1 = 1.0025e-05;
	double x2 = 1.002e-05;
	eps_less<double> lessthan;
	if (lessthan(x1, x2)) {
		cout << "epsilon wrong\n";
	} else {
		cout << "correct\n";
	}
#endif
}
}
