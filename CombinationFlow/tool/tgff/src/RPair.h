// Copyright 2017 by Robert Dick.
// All rights reserved.

#ifndef R_PAIR_H_
#define R_PAIR_H_

/* Defines multi-item objects and support routines.  Exposing the data for
these simple types is intentional. */

/*###########################################################################*/
#include "RFunctional.h"
#include "Interface.h"

#include <utility>
#include <iosfwd>

namespace rstd {
/*===========================================================================*/
/* Don't even think of providing 'ostream << pair'.  It separates the
definition of pair's interface into two files and may result in serious
problems later. */

template <typename Pair>
	struct equal_to_1st : public rbinary_function<Pair, Pair, bool>
	{ bool operator()(const Pair & a, const Pair & b) const; };

template <typename Pair>
	struct less_1st : public rbinary_function<Pair, Pair, bool>
	{ bool operator()(const Pair & a, const Pair & b) const; };

template <typename Pair>
	struct equal_to_2nd : public rbinary_function<Pair, Pair, bool>
	{ bool operator()(const Pair & a, const Pair & b) const; };

template <typename Pair>
	struct less_2nd : public rbinary_function<Pair, Pair, bool>
	{ bool operator()(const Pair & a, const Pair & b) const; };

template <typename Pair>
	struct select1st :
	public runary_function<Pair, typename Pair::first_type>
	{ const typename Pair::first_type & operator()(const Pair & a) const; };

template <typename Pair>
	struct select2nd :
	public runary_function<Pair, typename Pair::second_type>
	{ const typename Pair::second_type & operator()(const Pair & a) const; };

/*===========================================================================*/
template <typename T1, typename T2, typename T3>
struct Triple : public Prints<Triple<T1, T2, T3> > {
	typedef T1 first_type;
	typedef T2 second_type;
	typedef T3 third_type;

	~Triple() {}
	Triple() : first(T1()), second(T2()), third(T3()) {}
	Triple(const T1 & a, const T2 & b, const T3 & c);

	template <typename TripleLike>
		Triple(const TripleLike & tl);

	void print_to(std::ostream & os) const;

		T1 first;
		T2 second;
		T3 third;
};

template <typename T1, typename T2, typename T3>
	bool operator<(const Triple<T1, T2, T3> & x,
	const Triple<T1, T2, T3> & y);

template <typename T1, typename T2, typename T3>
	bool operator==(const Triple<T1, T2, T3> & x,
	const Triple<T1, T2, T3> & y);

template <typename T1, typename T2, typename T3>
	Triple<T1, T2, T3> make_triple(const T1 & x, const T2 & y,
	const T3 & z);

template <typename Triple>
	struct select3rd :
	public runary_function<Triple, typename Triple::third_type>
	{ const typename Triple::third_type & operator()(const Triple & a) const; };

template <typename Triple>
	struct equal_to_3rd : public rbinary_function<Triple, Triple, bool>
	{ bool operator()(const Triple & a, const Triple & b) const; }; 

template <typename Triple>
	struct less_3rd : public rbinary_function<Triple, Triple, bool>
	{ bool operator()(const Triple & a, const Triple & b) const; }; 

/*###########################################################################*/
template <typename T1, typename T2, typename T3, typename T4>
struct Quad : public Prints<Quad<T1, T2, T3, T4> > {
	typedef T1 first_type;
	typedef T2 second_type;
	typedef T3 third_type;
	typedef T4 fourth_type;

	~Quad() {}

	Quad() : first(T1()), second(T2()), third(T3()), fourth(T4()) {}
	Quad(const T1 & a, const T2 & b, const T3 & c, const T4 & d);

	template <typename QuadLike>
		Quad(const QuadLike & ql);

	void print_to(std::ostream & os) const;

		T1 first;
		T2 second;
		T3 third;
		T4 fourth;
};

template <typename T1, typename T2, typename T3, typename T4>
	bool operator<(const Quad<T1, T2, T3, T4> & x,
	const Quad<T1, T2, T3, T4> & y);

template <typename T1, typename T2, typename T3, typename T4>
	bool operator==(const Quad<T1, T2, T3, T4> & x,
	const Quad<T1, T2, T3, T4> & y);

template <typename T1, typename T2, typename T3, typename T4>
	Quad<T1, T2, T3, T4> make_quad(const T1 & x, const T2 & y,
	const T3 & z, const T4 & p);

template <typename Quad>
	struct select4th : public runary_function<Quad, typename Quad::fourth_type>
	{ const typename Quad::fourth_type & operator()(const Quad & a) const; };

template <typename Quad>
	struct equal_to_4th : public rbinary_function<Quad, Quad, bool>
	{ bool operator()(const Quad & a, const Quad & b) const; };

template <typename Quad>
	struct less_4th : public rbinary_function<Quad, Quad, bool>
	{ bool operator()(const Quad & a, const Quad & b) const; }; 

/*###########################################################################*/
#include "RPair.cct"
}

#endif
