// Copyright 2017 by Robert Dick.
// All rights reserved.

#include "RMath.h"

#include "Epsilon.h"
#include "RAlgo.h"
#include "RGen.h"
#include "RString.h"

#include <typeinfo>
#include <cstddef>
#include <set>
#include <iostream>

/*###########################################################################*/
namespace rstd {
using namespace std;

namespace{
unsigned gcd(unsigned m, unsigned n) {
	unsigned t;

	do {
		t = n % m;
		n = m;
		m = t;
	} while (m);
	return n;
}

bool finite(double x) {
	return x <= numeric_limits<double>::max() &&
		x >= -numeric_limits<double>::max();
}
}

/*===========================================================================*/
double pdf_triangle(double urv01_a, double urv01_b, double slope) {
	Rassert(slope >= 0.0 && slope <= 1.0);

	slope = slope / 2.0 + 0.5;
	double g = abs(urv01_a - 0.5 + urv01_b - 0.5) / slope;

	if (g >= 1.0) {
// Inverse map the tail back on itself.
		double range = 1.0 / slope - 1;
		g = 1.0 - (g - 1.0) / range;
	}

// Clean it up.
	g = max(g, 0.0);
	g = min(g, 1.0 - numeric_limits<double>::epsilon());
	RASSERT(g < 1.0);

	return g;
}

/*===========================================================================*/
static double lcm(double i, double j, double ref) {
	if (eps_is_equal_to(i, j, ref)) {
		return i;
	}

	RASSERT(i < j);
	unsigned i_mul = 1;
	unsigned j_mul = 2;

	double i_prod = i;
	double j_prod = j;

	do {
		if (i_prod < j_prod) {
			unsigned old_i_mul = i_mul;
			i_mul = static_cast<unsigned>(floor(j_prod / i));
			if (old_i_mul == i_mul) ++i_mul;
			i_prod = i_mul * i;
		} else {
			unsigned old_j_mul = j_mul;
			j_mul = static_cast<unsigned>(floor(i_prod / j));
			if (old_j_mul == j_mul) ++j_mul;
			j_prod = j_mul * j;
		}
	} while (! eps_is_equal_to(i_prod, j_prod, ref));

	return i_prod;
}

/*===========================================================================*/
static inline double gcd(double m, double n, double ref) {
	return (m * n) / lcm(m, n, ref);
}

/*===========================================================================*/
static inline unsigned lcm(unsigned i, unsigned j) {
	return (i * j) / gcd(i, j);
}

/*===========================================================================*/
unsigned lcm(const RVector<unsigned> & num) {
	set<unsigned> srt(num.begin(), num.end());

	while (srt.size() > 1) {
		unsigned a = * srt.begin();
		set<unsigned>::iterator beg = srt.begin();
		srt.erase(beg);

		unsigned b = * srt.begin();
		srt.erase(srt.begin());

		unsigned c = lcm(a, b);
		srt.insert(c);
	}

	return * srt.begin();
}

/*===========================================================================*/
double lcm(const RVector<double> & num, double ref) {
	set<double> srt(num.begin(), num.end());

	while (srt.size() > 1) {
		double a = * srt.begin();
		set<double>::iterator beg = srt.begin();
		srt.erase(beg);

		double b = * srt.begin();
		srt.erase(srt.begin());

		double c = lcm(a, b, ref);
		srt.insert(c);
	}

	return * srt.begin();
}

/*===========================================================================*/
unsigned gcd(const RVector<unsigned> & num) {
	set<unsigned> srt(num.begin(), num.end());

	while (srt.size() > 1) {
		unsigned a = * srt.begin();
		srt.erase(srt.begin());

		unsigned b = * srt.begin();
		srt.erase(srt.begin());

		unsigned c = gcd(a, b);
		srt.insert(c);
	}

	return * srt.begin();
}

/*===========================================================================*/
double gcd(const RVector<double> & num, double ref) {
	set<double> srt(num.begin(), num.end());

	while (srt.size() > 1) {
		double a = * srt.begin();
		srt.erase(srt.begin());

		double b = * srt.begin();
		srt.erase(srt.begin());

		double c = gcd(a, b, ref);
		srt.insert(c);
	}

	return * srt.begin();
}

/*===========================================================================*/
double
flat_merge(double alpha, const double a, const double b) {
/* Takes the integral of the convolution of the PDFs of the URV probability
distributions of a and b and maps the output back to a URV after a weighted
combination.  See written documentation. */
	RASSERT(alpha >= 0.0 && alpha <= 1.0);
	double beta = 1.0 - alpha;

	if (alpha > beta) swap(alpha, beta);

	RASSERT(a >= -1.0 && a <= 1.0);
	RASSERT(b >= -1.0 && b <= 1.0);

	double raw_merge = alpha * a + beta * b;
	const double neg = raw_merge < 0.0 ? -1.0 : 1.0;

	raw_merge = fabs(raw_merge);

	const double height = 1.0 / (1.0 - alpha);

	double mrg = numeric_limits<double>::max();

	if (raw_merge <= 1.0 - 2.0 * alpha) {
		mrg = height * raw_merge;
	} else {
		const double inner = (raw_merge - 1.0) * (raw_merge - 1.0);

		mrg = height * alpha - (height * inner) / (4.0 * alpha) +
			height * (1.0 - 2.0 * alpha);
	}

	RASSERT(mrg >= 0.0 && mrg <= 1.0);
	return mrg * neg;
}

/*###########################################################################*/
bool boltzmann_trial(double old_rank, double new_rank, double temperature,
RGen & rgen) {
	double cut = rgen.flat_range_d(0.0, 1.0);

	if (! temperature) {
		return new_rank > old_rank;
	}

	RASSERT(temperature);
	return 1.0 / (1.0 + exp((new_rank - old_rank) / temperature)) < cut;
}

/*===========================================================================*/
bool boltzmann_trial(double old_rank, double new_rank, double temperature) {
	return boltzmann_trial(old_rank, new_rank, temperature, RGen::gen());
}

/*===========================================================================*/
bool boltzmann_trial_range01(double old_rank, double new_rank,
double temperature01, RGen & rgen) {
	RASSERT(temperature01 >= 0.0);

	const double temp_0_inf = temperature01 ?
		(1.0 / (1.0 - temperature01)) - 1.0: numeric_limits<double>::max();

	return boltzmann_trial(old_rank, new_rank, temp_0_inf, rgen);
}

/*===========================================================================*/
bool boltzmann_trial_range01(double old_rank, double new_rank,
double temperature01) {
	return boltzmann_trial_range01(old_rank, new_rank, temperature01,
		RGen::gen());
}

/*###########################################################################*/
MathVec
MathVec::operator-=(const MathVec & a) {
	MAP(x, size()) (*this)[x] -= a[x];
	return *this;
}

/*===========================================================================*/
MathVec
MathVec::operator-=(const value_type & a) {
	MAP(x, size()) (*this)[x] -= a;
	return *this;
}

/*===========================================================================*/
MathVec
MathVec::operator+=(const MathVec & a) {
	MAP(x, size()) (*this)[x] += a[x];
	return *this;
}

/*===========================================================================*/
MathVec
MathVec::operator+=(const value_type & a) {
	MAP(x, size()) (*this)[x] += a;
	return *this;
}

/*===========================================================================*/
MathVec
MathVec::operator/=(const MathVec & a) {
	MAP(x, size()) (*this)[x] /= a[x];
	return *this;
}

/*===========================================================================*/
MathVec
MathVec::operator/=(const value_type & a) {
	MAP(x, size()) (*this)[x] /= a;
	return *this;
}

/*===========================================================================*/
MathVec &
MathVec::flat_mul(const MathVec & a) {
	MAP(x, size()) (*this)[x] *= a[x];
	return *this;
}

/*===========================================================================*/
MathVec &
MathVec::flat_mul(const value_type & a) {
	MAP(x, size()) (*this)[x] *= a;
	return *this;
}

/*===========================================================================*/
MathVec &
MathVec::random_dir(value_type magnitude, RGen & rg) {
	bool all_zero = true;
	MAP(x, size()) {
		(*this)[x] = rg.gauss_mean_d(0.0, 1.0);
		if ((*this)[x])
			all_zero = false;
	}

	if (all_zero) {
// Do it the slow but certain way.
		MAP(x, size()) {
			if (rg.flip()) {
				(*this)[x] = max(numeric_limits<double>::min(),
					rg.gauss_mean_half_d(0.0, 1.0));
			} else {
				(*this)[x] = min(-numeric_limits<double>::min(),
					-rg.gauss_mean_half_d(0.0, 1.0));
			}
		}
	}

	normalize(magnitude);

	return * this;
}

/*===========================================================================*/
MathVec &
MathVec::random_dir(value_type magnitude) {
	return random_dir(magnitude, RGen::gen());
}

/*===========================================================================*/
MathVec &
MathVec::random_pos(value_type radius, RGen & rg) {
	double ln = rg.flat_range_d(0.0, 1.0);
	return random_dir(pow(ln, static_cast<double>(size()) - 1.0) * radius);
}

/*===========================================================================*/
MathVec &
MathVec::random_pos(value_type radius) {
	return random_pos(radius, RGen::gen());
}

/*===========================================================================*/
MathVec &
MathVec::normalize(value_type magnitude) {
	value_type this_mag = mag();

	value_type m = magnitude / this_mag;

	if (finite(m)) {
		MAP(i, size()) (*this)[i] *= m;
	} else {
		string printed;
		MAP(x, size()) {
			printed += to_string((*this)[x]);
			if (x < size() - 1) printed += " ";
		}
		throw underflow_error(printed);
	}

#ifdef ROB_DEBUG
	MAP(i, size()) {
		Rassert(finite((*this)[i]));
	}
#endif

	return *this;
}

/*===========================================================================*/
MathVec &
MathVec::clip(value_type low, value_type high) {
	MAP(x, size())
		(*this)[x] = rstd::clip((* this)[x], low, high);
	return *this;
}

/*===========================================================================*/
MathVec::value_type
MathVec::mag() const {
	value_type accum = 0.0;
	MAP(x, size())
		accum += (*this)[x] * (*this)[x];

	RASSERT(finite(accum));
	RASSERT(accum >= 0);

	accum = sqrt(accum);
	RASSERT(finite(accum));

	return accum;
}

/*===========================================================================*/
MathVec::value_type
MathVec::operator*(const MathVec & a) const {
	RASSERT(size() == a.size());
	MathVec::value_type accum = 0.0;
	MAP(x, size()) {
		accum += (*this)[x] * a[x];
	}

	return accum;
}

/*===========================================================================*/
const RVector<MathVec::value_type>
operator*(const RVector<MathVec> & a,
const MathVec & b) {
	RVector<MathVec::value_type> vec(a.size());
	MAP(x, a.size()) {
		vec[x] = a[x] * b;
	}
	return vec;
}
 
/*===========================================================================*/
const MathVec
max_dimen(RVector<MathVec> & vec) {
	MathVec max_v(vec[0].size(), -numeric_limits<double>::max());
	MAP2(x, vec.size(), y, vec[x].size())
		max_v[y] = max(max_v[y], vec[x][y]);
	return max_v;
}

/*===========================================================================*/
const MathVec
min_dimen(RVector<MathVec> & vec) {
	MathVec min_v(vec[0].size(), numeric_limits<double>::max());
	MAP2(x, vec.size(), y, vec[x].size())
		min_v[y] = min(min_v[y], vec[x][y]);
	return min_v;
}

/*===========================================================================*/
void normalize(RVector<MathVec> & vec) {
	if (vec.empty())
		return;

	MathVec max_v = max_dimen(vec);
	MathVec min_v = min_dimen(vec);

	MathVec half_range = (max_v - min_v) / 2.0;
	MathVec mhr = min_v + half_range;

	MathVec::value_type max_mag = 0.0;
	MAP(x, vec.size())
		max_mag = max(max_mag, vec[x].mag());

	MAP(x, vec.size())
		vec[x] = (vec[x] - mhr) / max_mag;
}

/*===========================================================================*/
double
lambert(const double z) {
/* Lambert W function. 
Based on code written by K M Briggs.
http://www.btexact.com/people/briggsk2/LambertW.c.
This method is not covered by the copyrights or protections which may apply to
the rest of the code. */

	const double eps = 4.0e-16;
	const double em1 = 0.3678794411714423215955237701614608; 

	RASSERT(z >= -em1 && finite(z));

	if (! z) {
		return 0.0;
	}

// Cache the last value returned.
	static pair<double, double> cache;

	pair<double, double> current_cache = cache;
	if (eps_is_equal_to(z, current_cache.first, eps)) {
		return current_cache.second;
	}

// series near -em1 in sqrt(q)
	double w;

	if (z < -em1 + 1e-4) {
		double q = z + em1;
		double r = sqrt(q);
		double q2 = q * q;
		double q3 = q2 * q;

// error approx 1e-16
		w = -1.0 +
			2.331643981597124203363536062168 * r -
			1.812187885639363490240191647568 * q +
			1.936631114492359755363277457668 * r * q -
			2.353551201881614516821543561516 * q2 +
			3.066858901050631912893148922704 * r * q2 -
			4.175335600258177138854984177460 * q3 +
			5.858023729874774148815053846119 * r * q3 -
			8.401032217523977370984161688514 * q3 * q;

		cache = make_pair(z, w);
		return w;
	}

// initial approx for iteration...
// series near 0
	if (z < 1.0) {
		double p = sqrt(2.0 * (2.7182818284590452353602874713526625 * z + 1.0));
		w = -1.0 + p * (1.0 + p *
			(-0.333333333333333333333 + p * 0.152777777777777777777777));
	} else  {
// asymptotic
		w = log(z);
	}

	if (z > 3.0) {
// useful?
		w -= log(w);
	}

// Halley iteration
	for (long i = 0; i < 10; ++i) {
		double e = exp(w); 
		double dif = w * e - z;
		double p = w + 1.0;
		dif /= e * p - 0.5 * (p + 1.0) * dif / p; 
		w -= dif;

		if (fabs(dif) < eps * (1.0 + fabs(w))) {
// rel-abs error
			cache = make_pair(z, w);
			return w;
		}
	}

	Rabort();
	return 0.0;
}

/*===========================================================================*/
double
lambert_ddz(const double z) {
	double lz = lambert(z);
	return lz / (z * (1.0 + lz));
}

/*###########################################################################*/
void RMath_test() {
	Rassert(boltzmann_trial(0.0, 1.0, 0.0, RGen::gen()));
	Rassert(! boltzmann_trial(1.0, 0.0, 0.0, RGen::gen()));

	for (long x = -50; x < 50; ++x) {
		double y = static_cast<double>(x);
		Rassert(interval_round(x, 7) == interval_round(y, 7.0));
	}

//	double k = lcm(1.0, 0.29999999999999999);
}
}
