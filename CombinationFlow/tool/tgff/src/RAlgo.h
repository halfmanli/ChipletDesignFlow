// Copyright 2017 by Robert Dick.
// All rights reserved.

#ifndef R_ALGO_H_
#define R_ALGO_H_

// Provides simple generic algorithms.

/*###########################################################################*/
#include "RFunctional.h"
#include "RVector.h"

#include <algorithm>
#include <utility>
#include <iosfwd>
#include <cstdlib>
#include <cmath>

namespace rstd {

/*###########################################################################*/
class SearchHalt {
public:
	SearchHalt() : val_(0) {}
	SearchHalt(long x) : val_(x) {}
	operator long() const { return val_; }
	SearchHalt & operator|=(const SearchHalt sh);

// Target higher than search range.
	static const long RANGE_LOW = 0x1;
// Target lower than search range.
	static const long RANGE_HIGH = 0x2;
// Too many iterations.
	static const long TIMEOUT = 0x4;
// Noise encountered.  Closest match returned.
	static const long NOISE = 0x8;
// Found a match.
	static const long MATCH = 0x10;
// Precise enough.
	static const long GOOD = 0x40;

private:
		long val_;
};

std::ostream & operator<<(std::ostream & os, const SearchHalt & sh);

/* Continuous binary search for x s.t. func(x) == target between low and high.
Function must be non-increasing or non-decreasing.  func(low) < func(high)
Returns a false if the target was out of range.  */

template <typename T, typename FUNC, typename EQ>
	std::pair<SearchHalt, T> chop_search(T target, T low, T high, FUNC func,
	EQ eq, T precision = 1E16, long timeout = 10000);

// Discrete binary search. 

template <typename T, typename FUNC>
	std::pair<SearchHalt, long> int_chop_search(T target, long low, long high,
	FUNC func, T precision = 1E16);

template <typename T, typename FUNC, typename INC>
	T deriv(T x, FUNC func, INC inc);

// STL extension
template<typename I, typename T>
	void iota(I first, I last, T val);

// STL extension
template<typename I, typename O, typename D>
O random_sample_n(I first, I last, O out, const D n);

// SGI extension
template<typename I, typename O, typename D, typename G>
O random_sample_n(I first, I last, O out, const D n, G gen);

/*===========================================================================*/
// Puts indexes of elements in [begin, end), in increasing order, to dest.
template <typename InIter, typename OutIter>
	OutIter pri_map(InIter begin, const InIter end,
	OutIter dest);

/*===========================================================================*/
// Resizes c1 in c2's shape, filling it with value.  C1 must support resize().

// Resizes a container from another container.

// First two for completeness.
template <typename C1, typename C2, typename T>
	void resize_from(C1 & c1, const C2 & c2, const T & value);

template <typename C1, typename C2>
	void resize_from(C1 & c1, const C2 & c2);

template <typename C1, typename C2, typename T>
	void resize2_from(C1 & c1, const C2 & c2, const T & value);

template <typename C1, typename C2>
	void resize2_from(C1 & c1, const C2 & c2);

template <typename C1, typename C2, typename T>
	void resize3_from(C1 & c1, const C2 & c2, const T & value);

template <typename C1, typename C2>
	void resize3_from(C1 & c1, const C2 & c2);

template <typename C1, typename C2, typename T>
	void resize4_from(C1 & c1, const C2 & c2, const T & value);

template <typename C1, typename C2>
	void resize4_from(C1 & c1, const C2 & c2);

/*===========================================================================*/
// Resizes based on container contents.
template <typename C1, typename C2, typename T>
	void resize2_from_deref(C1 & c1, const C2 & c2, const T & value);

template <typename C1, typename C2>
	void resize2_from_deref(C1 & c1, const C2 & c2);

template <typename C1, typename C2, typename T>
	void resize3_from_deref(C1 & c1, const C2 & c2, const T & value);

template <typename C1, typename C2>
	void resize3_from_deref(C1 & c1, const C2 & c2);

template <typename C1, typename C2, typename T>
	void resize4_from_deref(C1 & c1, const C2 & c2, const T & value);

template <typename C1, typename C2>
	void resize4_from_deref(C1 & c1, const C2 & c2);

/*===========================================================================*/
// Resizes to rectangle based on dimension arguments.

// First two for completeness.
template <typename C, typename T>
	void resize(C & con, unsigned a, const T & value);

template <typename C>
	void resize(C & con, unsigned a);

template <typename C, typename T>
	void resize2(C & con, unsigned a, unsigned b, const T & value);

template <typename C>
	void resize2(C & con, unsigned a, unsigned b);

template <typename C, typename T>
	void resize3(C & con, unsigned a, unsigned b, unsigned c, const T & value);

template <typename C>
	void resize3(C & con, unsigned a, unsigned b, unsigned c);

template <typename C, typename T>
	void resize4(C & con, unsigned a, unsigned b, unsigned c, unsigned d,
	const T & value);

template <typename C>
	void resize4(C & con, unsigned a, unsigned b, unsigned c, unsigned d);

template <typename ITER, typename T>
	void set_vec(ITER i, const T & a, const T & b);

template <typename ITER, typename T>
	void set_vec(ITER i, const T & a, const T & b, const T & c);

template <typename ITER, typename T>
	void set_vec(ITER i, const T & a, const T & b, const T & c, const T & d);

/*###########################################################################*/
#include "RAlgo.cct"
}
#endif
