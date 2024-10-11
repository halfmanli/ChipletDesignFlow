// Copyright 2017 by Robert Dick.
// All rights reserved.

#ifndef EPSILON_H_
#define EPSILON_H_

// Epsilon computation.  For now, only works for floats and doubles.

/*###########################################################################*/
#include <limits>
#include <cmath>

#ifdef ROB_DEBUG
#	include "RStd.h"
#endif

#include "RFunctional.h"
#include "Interface.h"

/* REF_EXP is used to provide thresholds when comparing operands that are now
small but were the result of operations on larger values.  It should the number
of zeros after 1 for the maximum values used in the production of the
operands. */

namespace rstd {
/*###########################################################################*/
template <typename T>
	bool eps_is_equal_to(const T & a, const T & b, const T & ref);

template <typename T>
	bool eps_is_less(const T & a, const T & b, const T & ref);

/*===========================================================================*/
template <typename T, int REF_EXP = -500>
struct eps_equal_to : rbinary_function<const T &, const T &, bool> {
	eps_equal_to();
	bool operator()(const T & a, const T & b) const;
		T ref_;
};

/*===========================================================================*/
template <typename T, int REF_EXP = -500>
struct eps_less : rbinary_function<const T &, const T &, bool> {
	eps_less();
	bool operator()(const T & a, const T & b) const;
		T ref_;
};

/*===========================================================================*/
template <typename T, int REF_EXP = -500>
struct eps_not_equal_to : rbinary_function<const T &, const T &, bool> {
	eps_not_equal_to();
	bool operator()(const T & a, const T & b) const;
		T ref_;
};

template <typename T, int REF_EXP = -500>
struct eps_greater : rbinary_function<const T &, const T &, bool> {
	eps_greater();
	bool operator()(const T & a, const T & b) const;
		T ref_;
};

template <typename T, int REF_EXP = -500>
struct eps_less_equal : rbinary_function<const T &, const T &, bool> {
	eps_less_equal();
	bool operator()(const T & a, const T & b) const;
		T ref_;
};

template <typename T, int REF_EXP = -500>
struct eps_greater_equal : rbinary_function<const T &, const T &, bool> {
	eps_greater_equal();
	bool operator()(const T & a, const T & b) const;
		T ref_;
};

/*===========================================================================*/
template <typename T>
struct eps_inc : runary_function<const T &, T> {
	eps_inc(long step);
	void operator()(T & a) const;

	T step_;
};

/*===========================================================================*/
template <typename T>
struct eps_dec : runary_function<const T &, T> {
	eps_dec(long step);
	void operator()(T & a) const;

	T step_;
};

/*###########################################################################*/
template <typename T, int REF_EXP = -500>
class eps_comp_obj {
public:
	eps_comp_obj() : lt_(eps_less<T, REF_EXP>()) {}
	comp_type operator()(const T & a, const T & b) const;
private:
		eps_less<T, REF_EXP> lt_;
};

/*===========================================================================*/
void Epsilon_test();

/*###########################################################################*/
#include "Epsilon.cct"
}
#endif

