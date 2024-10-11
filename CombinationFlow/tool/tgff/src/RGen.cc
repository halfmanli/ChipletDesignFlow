// Copyright 2017 by Robert Dick.
// All rights reserved.

#include "RGen.h"

#include "RMath.h"
#include "RStd.h"
#include "Epsilon.h"
#include "RVector.h"

#include <cmath>
#include <cfloat>
#include <iostream>
#include <iomanip>
#include <memory>

#include "RGenData.h"

namespace rstd {
using namespace std;

/*############################################################################
Robert Dick (dickrp@wckn.dorm.clarkson.edu)

This is an implementation of the universal random number generator
proposed by G. Marsaglia and A. Zaman in
	G. Marsaglia, A. Zaman, W. Tseng
	Statistics and Probability Letters
	1990, volume 9, number 1, page 35.

The generation of i, j, k, and l from one seed is done using the
method proposed in
	F. James
	Computer Physics Communications
	1990, volume 60, page 329

Portable to machines with 32 bit floating point numbers with 24 bit mantissas.
Note that only the most significant 24 bits of the mantissa are guaranteed to
be identical from machine to machine.

Gaussian RVs are generated using the Box-Muller transformation.  This is not
an approximation like the 12 sum approach.  True Gaussian RVs are generated.
############################################################################*/

/*##########################################################################*/
RGen::~RGen() {}

/*===========================================================================*/
RGen::RGen(int seed) :
	ip_(0), jp_(0),
	c_(0), cd_(0), cm_(0),
	u_(new RVector<float>(97))
{
	set_seed(seed);
}

/*===========================================================================*/
void RGen::rswap(self & a) {
	rstd::rswap(ip_, a.ip_);
	rstd::rswap(jp_, a.jp_);
	rstd::rswap(c_, a.c_);
	rstd::rswap(cd_, a.cd_);
	rstd::rswap(cm_, a.cm_);
	rstd::rswap(u_, a.u_);
}

/*===========================================================================*/
void RGen::set_seed(int seed) {
	int ij = seed / 30082;
	int kl = seed - 30082 * ij;

	int i = ij / 177 % 177 + 2;
	int j = ij % 177 + 2;
	int k = kl / 169 % 178 + 1;
	int l = kl % 169;

	set_seed(i, j, k, l);
}

/*===========================================================================*/
void RGen::set_seed(int i, int j, int k, int l) {
	MAP(ii, u_->size()) {
		float s = 0.0;
		float t = 0.5F;

		for (int jj = 0; jj < 24; jj++) {
			int m = (i * j % 179) * k % 179;

			i = j; j = k; k = m;

			l = (53 * l + 1) % 169;

			if (l * m % 64 >= 32)
				s += t;

			t *= 0.5F;
		}

		(*u_)[ii] = s;
	}

	c_ = 362436.0F / 16777216.0F;
	cd_ = 7654321.0F / 16777216.0F;
	cm_ = 16777213.0F / 16777216.0F;

	ip_ = 96;
	jp_ = 32;
}

/*===========================================================================*/
double RGen::flat01() {
	float uni = (*u_)[ip_] - (*u_)[jp_];
	if (uni < 0.0)
		uni += 1.0F;

	(*u_)[ip_] = uni;
	ip_ ? ip_-- : ip_ = 96;
	jp_ ? jp_-- : jp_ = 96;
	c_ -= cd_;

	if (c_ < 0.0) c_ += cm_;

	uni -= c_;
	if (uni < 0.0) uni += 1.0F;

	RASSERT(uni < 1.0F);

	return uni;
}

/*===========================================================================*/
double RGen::gauss01() {
// Mean 0, variance 1.
		static int recalc=1;
		double in_a, in_b;
		double out_a;
		static double out_b;
		double modifier;
		double compile_b;

		if (recalc) {
// Range from (0:1], not [0:1).  Had to change this to prevent log(0).
			in_a = 1.0 - flat01();
			in_b = flat01();
		
			modifier = sqrt(-2.0 * log(in_a));
			compile_b = 2.0 * PI * in_b;
		
			out_a = modifier * cos(compile_b);
			out_b = modifier * sin(compile_b);

			recalc = 0;

			RASSERT(rstd::isfinite(out_a));
			return out_a;
		}

		recalc = 1;
		RASSERT(rstd::isfinite(out_b));
		return out_b;
}

/*===========================================================================*/
int RGen::operator()(int ceiling) {
	return int(ceiling * flat01());
}

/*===========================================================================*/
double RGen::flat_range_d(double low, double high) {
	RASSERT(low < high);
	return (high - low) * flat01() + low;
}

/*===========================================================================*/
long RGen::flat_range_l(long low, long high) {
	RASSERT(low < high);
	long val = static_cast<long>(static_cast<double>(high - low) * flat01()) + low;
	RASSERT(val >= low && val < high);
	return val;
}

/*===========================================================================*/
double RGen::gauss_mean_d(double mean, double variance) {
	double ret = gauss01() * variance + mean;	
	RASSERT(rstd::isfinite(ret));
	return ret;
}

/*===========================================================================*/
long RGen::gauss_mean_l(long mean, double variance) {
	return static_cast<long>(rint(gauss01() * variance + static_cast<double>(mean)));
}

/*===========================================================================*/
double RGen::gauss_mean_half_d(double mean, double variance) {
	return mean + variance * abs(gauss01());
}

/*===========================================================================*/
long RGen::gauss_mean_half_l(long mean, double variance) {
	return static_cast<long>(rint(static_cast<double>(mean) + variance * abs(gauss01())));
}

/*===========================================================================*/
bool RGen::flip(double cut_off) {
	return (flat01() < cut_off) ? 1 : 0;
}

/*===========================================================================*/
RGen &
RGen::gen() {
	static RGen g;
	return g;
}

/*===========================================================================*/
namespace {

void gen_confirm(double a, double b) {
// Do a floating point (not double) comparison.
// There's a limit to the precision and reproducibility.
	eps_equal_to<float> eq;

	if (! eq(static_cast<float>(a), static_cast<float>(b))) {
		cerr << "RGen: Test comparison error.\n" << setprecision(80) << a <<
			"\n!=\n" << setprecision(80) << b << "\n" << "difference: " <<
			setprecision(80) << (a - b) << "\n";

		Rabort();
	}	
}

}

void RGen_test() {
// RGen::gen_test_data(cout);

#ifdef ROB_DEBUG
	RGen g;

	g.set_seed(0);
	MAP(x, 25)
		gen_confirm(g.flat_range_d(-1024, 20), L_data1[x]);

	g.set_seed(1);
	MAP(x, 25)
		gen_confirm(g.flat_range_d(-10, 2048), L_data2[x]);

	g.set_seed(3);
	MAP(x, 25)
		gen_confirm(g.flat_range_d(234, 22523), L_data3[x]);

	g.set_seed(51);
	MAP(x, 25)
		gen_confirm(g.flat_range_d(-1024, 2048), L_data4[x]);

	g.set_seed(1024);
	MAP(x, 25)
		gen_confirm(g.flat_range_d(-51231, -123), L_data5[x]);

	g.set_seed(0);
	MAP(x, 25)
		gen_confirm(g.gauss_mean_d(-1024, 2048), L_data6[x]);

	g.set_seed(1);
	MAP(x, 25)
		gen_confirm(g.gauss_mean_d(-1024, 2048), L_data7[x]);

	g.set_seed(3);
	MAP(x, 25)
		gen_confirm(g.gauss_mean_d(-1024, 2048), L_data8[x]);

	g.set_seed(51);
	MAP(x, 25)
		gen_confirm(g.gauss_mean_d(-1024, 2048), L_data9[x]);

	g.set_seed(1024);
	MAP(x, 25)
		gen_confirm(g.gauss_mean_d(-1024, 2048), L_data10[x]);
#endif

#if 0
	long cnt = 0;
	double var_sum = 0.0;

	while (1) {
		double x = RGen::gen.flat_range_d(0.0, 1.0);
		var_sum += (0.5 - x) * (0.5 - x);
		cnt++;

		cout << var_sum / cnt << "\t" << pow(var_sum / cnt, 0.5) << "\n";
	}
#endif

#if 0
	long a = 0, b = 0, c = 0, d = 0, e = 0, f = 0, g = 0,
	  h = 0, i = 0, j = 0;

	double total = 0.0;

	long ticker = 0;

	while (1) {		
		double fred = RGen::gen.flat_range_d(0.0, 1.0) +
		  RGen::gen.flat_range_d(0.0, 1.0) +
		  RGen::gen.flat_range_d(0.0, 1.0) +
		  RGen::gen.flat_range_d(0.0, 1.0) +
		  RGen::gen.flat_range_d(0.0, 1.0) +
		  RGen::gen.flat_range_d(0.0, 1.0) +
		  RGen::gen.flat_range_d(0.0, 1.0) +
		  RGen::gen.flat_range_d(0.0, 1.0) +
		  RGen::gen.flat_range_d(0.0, 1.0) +
		  RGen::gen.flat_range_d(0.0, 1.0);

		if (fred < 1.0)
			a++;
		else if (fred < 2.0)
			b++;
		else if (fred < 3.0)
			c++;
		else if (fred < 4.0)
			d++;
		else if (fred < 5.0)
			e++;
		else if (fred < 6.0)
			f++;
		else if (fred < 7.0)
			g++;
		else if (fred < 8.0)
			h++;
		else if (fred < 9.0)
			i++;
		else
			j++;

		total += 1.0;

		if (ticker == 1000) {
			cout << setw(10) << a / total << "\t" <<
			  setw(10) << b / total << "\t" <<
			  setw(10) << c / total << "\t" <<
			  setw(10) << d / total << "\t" <<
			  setw(10) << e / total << "\t" <<
			  setw(10) << f / total << "\t" <<
			  setw(10) << g / total << "\t" <<
			  setw(10) << h / total << "\t" <<
			  setw(10) << i / total << "\t" <<
			  setw(10) << j / total << "\n";	
		}
		ticker++;
	}
#endif
}

/*===========================================================================*/
void RGen::gen_test_data(ostream & os) {
	RGen g;

	os << setprecision(80);

	os << "#ifdef ROB_DEBUG\n\n"
		"namespace {\n\n"
		"double L_data1[] = {\n";

	g.set_seed(0);

	MAP(x, 25) {
		os << g.flat_range_d(-1024, 20);
		if (x < 24)
			os << ",\n";
	}

	os << "\n};\n\n" <<
		"static double L_data2[] = {\n";

	g.set_seed(1);
	MAP(x, 25) {
		os << g.flat_range_d(-10, 2048);
		if (x < 24)
			os << ",\n";
	}

	os << "\n};\n\n" <<
		"static double L_data3[] = {\n";

	g.set_seed(3);
	MAP(x, 25) {
		os << g.flat_range_d(234, 22523);
		if (x < 24)
			os << ",\n";
	}

	os << "\n};\n\n" <<
		"static double L_data4[] = {\n";

	g.set_seed(51);
	MAP(x, 25) {
		os << g.flat_range_d(-1024, 2048);
		if (x < 24)
			os << ",\n";
	}

	os << "\n};\n\n" <<
		"static double L_data5[] = {\n";

	g.set_seed(1024);
	MAP(x, 25) {
		os << g.flat_range_d(-51231, -123);
		if (x < 24)
			os << ",\n";
	}

	os << "\n};\n\n" <<
		"static double L_data6[] = {\n";

	g.set_seed(0);
	MAP(x, 25) {
		os << g.gauss_mean_d(-1024, 2048);
		if (x < 24)
			os << ",\n";
	}

	os << "\n};\n\n" <<
		"static double L_data7[] = {\n";

	g.set_seed(1);
	MAP(x, 25) {
		os << g.gauss_mean_d(-1024, 2048);
		if (x < 24)
			os << ",\n";
	}

	os << "\n};\n\n" <<
		"static double L_data8[] = {\n";

	g.set_seed(3);
	MAP(x, 25) {
		os << g.gauss_mean_d(-1024, 2048);
		if (x < 24)
			os << ",\n";
	}

	os << "\n};\n\n" <<
		"static double L_data9[] = {\n";

	g.set_seed(51);
	MAP(x, 25) {
		os << g.gauss_mean_d(-1024, 2048);
		if (x < 24)
			os << ",\n";
	}

	os << "\n};\n\n" <<
		"static double L_data10[] = {\n";

	g.set_seed(1024);
	MAP(x, 25) {
		os << g.gauss_mean_d(-1024, 2048);
		if (x < 24)
			os << ",\n";
	}

	os << "\n};\n\n"
		"}\n\n"
		"#endif\n";
}

}
