// Copyright 2017 by Robert Dick.
// All rights reserved.

#ifndef F_VECTOR_H_
#define F_VECTOR_H_

/*###########################################################################*/
#include "RStd.h"
#include "RAlgo.h"
#include "Interface.h"

#ifdef ROB_DEBUG
#	include <typeinfo>
#endif

#include <functional>
#include <cstddef>
#include <iterator>

namespace rstd {
/* Vector with size fixed at compile time.  It would be elegant to inherit
from RVector.  However, this turns out to be grotesquely inefficient.  In
order to get efficiency, some of the operations of this class are not
exception-safe.  Beware if you use this to hold anything but POD.

For FVectors holding two longs, it takes the vector implementation 60 times as
long to do a copy construction when compiled with gcc and optimized.  The
cross-over point lies sonewhere between 300 and 500 longs. There's a ~20% time
difference at 300, and no significant difference at 500.

A side-effect of this efficiency-driven implementation is the requirement that
T have a default constructor. */

/*===========================================================================*/
template <int N, typename T>
class FVector :
	public Prints<FVector<N, T>, T, true>,
	public Comps<FVector<N, T>, T, true>,
	public Clones<FVector<N, T> >,
	public Swaps<FVector<N, T> >,
	public SChecks<FVector<N, T> >
{
	typedef FVector<N, T> self;

public:
// Typedefs
	typedef T value_type;
	typedef T * pointer;
	typedef const T * const_pointer;
	typedef T * iterator;
	typedef const T * const_iterator;
	typedef T & reference;
	typedef const T & const_reference;
	typedef ptrdiff_t difference_type;
	typedef std::reverse_iterator<const_iterator> const_reverse_iterator;
	typedef std::reverse_iterator<iterator> reverse_iterator;	
	typedef long size_type;

// Construction
	virtual ~FVector();

	FVector() {
		static_assert(N, "Length-zero FVector.");
		RDEBUG(self_check());
	}

	explicit FVector(const T & value);

	template <typename InIter>
		FVector(InIter first, InIter last);

	FVector(const T & a, const T & b);
	FVector(const T & a, const T & b, const T & c);
	FVector(const T & a, const T & b, const T & c, const T & d);

// Interface
	virtual self * clone() const;
	virtual void rswap(self & a);
	virtual void self_check() const {}
	virtual void self_check_deep() const;

// Modifiable
	virtual self & operator=(const self & a);

// Final
	void print_to_default(std::ostream & os) const;
	comp_type comp_default(const self & b) const;
	iterator begin() { return data_; }
	const_iterator begin() const { return data_; }
	iterator end() { return data_ + N; }
	const_iterator end() const { return data_ + N; }

	iterator rbegin() { return reverse_iterator(data_ + N); }
	const_iterator rbegin() const;
	iterator rend() { return reverse_iterator(data_); }
	const_iterator rend() const;

	size_type size() const { return N; }

	void assign(size_type n, const T & val);

	template <class InIter>
		void assign(InIter first, InIter last);

	T & front() { return *data_; }
	const T & front() const { return *data_; }
	T & back() { return data_[N - 1]; }
	const T & back() const { return data_[N - 1]; }

	reference operator[](size_type x);
	const_reference operator[](size_type x) const;

private:
		T data_[N];

/* FIXME: Make this more strict when standard and compiler settle on
appropriate means of expressing constructor friendship and friend partial
specialization. */
	template <int, typename> friend class FVector;
};

/*===========================================================================*/
template <typename T>
	FVector<2, T> make_fvector(const T & a, const T & b);

template <typename T>
	FVector<3, T> make_fvector(const T & a, const T & b, const T & c);

template <typename T>
	FVector<4, T> make_fvector(const T & a, const T & b, const T & c,
	const T & d);

template <int DIM>
	bool increment(FVector<DIM, bool> & fv);

/*===========================================================================*/
void FVector_test();

/*###########################################################################*/
#include "FVector.cct"
}
#endif
