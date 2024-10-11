// Copyright 2017 by Robert Dick.
// All rights reserved.

#ifndef R_MATH_H_
#define R_MATH_H_

/*###########################################################################*/
#ifdef ROB_DEBUG
#	include "RStd.h"
#endif

#include "RVector.h"
#include "Epsilon.h"

#include <stdexcept>
#include <cstddef>
#include <iosfwd>
#include <limits>

namespace rstd {

class RGen;

/*###########################################################################*/
// Math utility functions.

/* Given two URVs [0,1), return a RV with a triangular PDF peaking
at 0.  At a slope of 1.0, it's a normal triangle.  At a slope of
0.0, it's a URV, */
double pdf_triangle(double urv01_a, double urv01_b, double slope = 1.0);

// Greatest common divisor.
unsigned gcd(const RVector<unsigned> & num);
double gcd(const RVector<double> & num, double ref = 0);

// Lowest common multiplier.
unsigned lcm(const RVector<unsigned> & num);
double lcm(const RVector<double> & num, double ref = 0);

// Ring modulus.
template <typename T, typename P> T ring_mod(T a, P period);
template <typename P> unsigned ring_mod(unsigned a, P period);
template <typename P> unsigned long ring_mod(unsigned long a, P period);
template <typename P> double ring_mod(double a, P period);
template <typename P> float ring_mod(float a, P period);

// Ring modulus support.
template <typename T, typename P> T interval_round(T a, P period);
template <typename P> double interval_round(double a, P period);
template <typename P> float interval_round(float a, P period);

// Clips to within a given range.
template<typename D> D clip(const D & x, D low, D high);

// Merges two weighted URVs into a URV.
double flat_merge(double alpha, double a, double b);

template <typename T> bool isfinite(const T & a);

/*===========================================================================*/
// Temperature range: [0:inf]
bool boltzmann_trial(double old_rank, double new_rank, double temperature,
	RGen & rgen);

bool boltzmann_trial(double old_rank, double new_rank, double temperature);

// Temperature range: [0:1]
bool boltzmann_trial_range01(double old_rank, double new_rank,
	double temperature01, RGen & rgen);

bool boltzmann_trial_range01(double old_rank, double new_rank,
	double temperature01);

/*===========================================================================*/
// Supports mathematical vector operations.

class MathVec : public RVector<double> {
	typedef RVector<double> super;
	typedef MathVec self;

public:
	MathVec() : super() {}
	MathVec(size_type n, const value_type & value) : super(n, value) {}
	MathVec(const RVector<value_type> & a) : super(a) {}
	explicit MathVec(size_type n) : super(n) {}

	template <typename InIter>
		MathVec(InIter first, InIter last) : super(first, last) {}

// Final
	MathVec operator+=(const MathVec &a);
	MathVec operator+=(const value_type &a);
	MathVec operator-=(const MathVec &a);
	MathVec operator-=(const value_type &a);
	MathVec operator/=(const MathVec &a);
	MathVec operator/=(const value_type &a);
	MathVec & flat_mul(const MathVec & a);
	MathVec & flat_mul(const value_type & a);
	MathVec & random_dir(value_type magnitude, RGen & rg);
	MathVec & random_dir(value_type magnitude = 1.0); 
	MathVec & random_pos(value_type radius, RGen & rg);
	MathVec & random_pos(value_type radius = 1.0);
/* Sets the length of the vector to magnitude.  Preserves direction.
Throws underflow_error if the magnitude of the vector is 0. */
	MathVec & normalize(value_type magnitude = 1.0);

	MathVec & clip(value_type low, value_type high);
	value_type mag() const;

	const MathVec operator+(const MathVec & b) const;
	const MathVec operator+(const MathVec::value_type & b) const;
	const MathVec operator-(const MathVec & b) const;
	const MathVec operator-(const MathVec::value_type & b) const;
	const MathVec operator/(const MathVec & b) const;
	const MathVec operator/(const MathVec::value_type & b) const;
	value_type operator*(const MathVec & b) const;
};

const MathVec flat_mul(const MathVec & a, const MathVec & b);

const MathVec flat_mul(const MathVec & a,
	const MathVec::value_type & b);

const RVector<MathVec::value_type>
	operator*(const RVector<MathVec> & a, const MathVec & b);

const MathVec max_dimen(RVector<MathVec> & vec);
const MathVec min_dimen(RVector<MathVec> & vec);
void normalize(RVector<MathVec> & vec);

double lambert(double z);
double lambert_ddz(double z);

template <typename T>
struct rnlimits : public std::numeric_limits<T> {
	typedef std::numeric_limits<T> super;
	static T smallest();
};

/*===========================================================================*/
void RMath_test();

/*###########################################################################*/
#include "RMath.cct"
}
#endif
