// Copyright 2017 by Robert Dick.
// All rights reserved.

#ifndef R_VECTOR_H_
#define R_VECTOR_H_

/*###########################################################################*/
#ifdef ROB_DEBUG
#	include <iostream>
#	include <typeinfo>
#endif

#include "RStd.h"
#include "Interface.h"

#include <iosfwd>
#include <vector>
#include <limits>
#include <functional>
#include <algorithm>

namespace rstd {

template <int, typename> class NoIndex;
template <int, typename> class FVector;

/*===========================================================================*/
/* A number of vector classes. */

/*===========================================================================*/
/* Implemented in terms of vector.  Changes vector's behavior.  Can't be
instantiated or destroyed.  For use in non-polymophic inheritance. */

template <typename T>
class RVector :
	public Prints<RVector<T>, T, true>,
	public Comps<RVector<T>, T, true>,
	public Clones<RVector<T> >,
	public Swaps<RVector<T> >,
	public SChecks<RVector<T> >
{
	typedef std::vector<T> impl;
	typedef RVector<T> self;

public:
// Changed.
	typedef long size_type;

// Typedefs
	typedef typename impl::value_type value_type;
	typedef typename impl::pointer pointer;
	typedef typename impl::const_pointer const_pointer;
	typedef typename impl::iterator iterator;
	typedef typename impl::const_iterator const_iterator;
	typedef typename impl::reference reference;
	typedef typename impl::const_reference const_reference;
	typedef typename impl::difference_type difference_type;
	typedef typename impl::allocator_type allocator_type;
	typedef typename impl::const_reverse_iterator const_reverse_iterator;
	typedef typename impl::reverse_iterator reverse_iterator;	

// Construction
	virtual ~RVector() {}
	RVector() : impl_() {}
	RVector(size_t n, const T & value) : impl_(n, value) {}
	explicit RVector(size_type n) : impl_(n) {}

	template <typename T2>
		RVector(const RVector<T2> & a) : impl_(a.begin(), a.end()) {}

	template <typename InIter>
		RVector(InIter first, InIter last) : impl_(first, last) {}

// Interface
	virtual void rswap(self & a) { impl_.swap(a.impl_); }
	virtual void self_check() const {}
	virtual void self_check_deep() const;
	virtual self * clone() const { return new self(*this); }

// Modifiable
	virtual self & operator=(const self & a);

// Final
	comp_type comp_default(const self & b) const;
	void print_to_default(std::ostream & os) const { print_cont(*this, os, " "); }
	size_type size() const { return static_cast<size_type>(impl_.size()); }
	size_type max_size() const { return std::numeric_limits<size_type>::max(); }
	T & operator[](size_type a);
	const T & operator[](size_type a) const;	

	iterator begin() { return impl_.begin(); }
	const_iterator begin() const { return impl_.begin(); }
	iterator end() { return impl_.end(); }
	const_iterator end() const { return impl_.end(); }
	reverse_iterator rbegin() { return impl_.rbegin(); }
	const_reverse_iterator rbegin() const { return impl_.rbegin(); }
	reverse_iterator rend() { return impl_.rend(); }
	const_reverse_iterator rend() const { return impl_.rend(); }
	size_type capacity() const { return impl_.capacity(); }
	bool empty() const { return impl_.empty(); }

	void assign(size_type n, const T & val) { impl_.assign(n, val); }

	template <class InIter>
		void assign(InIter first, InIter last);

	void reserve(size_type n) { impl_.reserve(n); }
	reference front() { return impl_.front(); }
	const_reference front() const { return impl_.front(); }
	reference back() { return impl_.back(); }
	const_reference back() const { return impl_.back(); }
	void push_back(const_reference x) { impl_.push_back(x); }
	void pop_back() { impl_.pop_back(); }

	iterator insert(iterator pos, const T & x);
	T * insert(iterator pos) { return impl_.insert(pos); }

	template <typename InIter>
		void insert(iterator pos, InIter first, InIter last);

	void insert(iterator pos, size_type n, const T & x);

	iterator erase(iterator a) { return impl_.erase(a); }
	iterator erase(iterator a, iterator b) { return impl_.erase(a, b); }
	void resize(size_type sz, const T & x) { impl_.resize(sz, x); }
	void resize(size_type new_size) { impl_.resize(new_size); }
	void clear() { impl_.clear(); }

private:
		impl impl_;
};

/*===========================================================================*/
/* Recursively defined multidimensional vector.
Use with INDEX of FVector for arbitrary indexing.
Use with INDEX inherited from FVector for specialized indexing.
Use with INDEX of NoIndex for no multidimensional indexing. */

template <int DIM, typename T,
template <int, typename> class INDEX = FVector>
class MVector :
	public RVector<MVector<DIM - 1, T, INDEX> >
{
	typedef RVector<MVector<DIM - 1, T, INDEX> > super;
	typedef MVector self;
	typedef typename super::size_type size_type;
	typedef typename super::reference reference;
	typedef typename super::const_reference const_reference;

public:
	virtual ~MVector() {}

	MVector() {}
	MVector(size_t n, typename super::const_reference value);
	explicit MVector(size_type n) : super(n) {}

	template <typename T2>
		MVector(const MVector<DIM, T2, INDEX> & a) : super(a) {}

	template <typename InIter>
		MVector(InIter first, InIter last) : super(first, last) {}

// New methods.
	template <typename IT>
		T & operator[](const INDEX<DIM, IT> & indx);

	template <typename IT>
		const T & operator[](const INDEX<DIM, IT> & indx) const;

// Forwarded methods.
	reference operator[](size_type a) { return super::operator[](a); }
	const_reference operator[](size_type a) const;

private:
	template <typename IT>
		T & dim_index(const IT * i);

	template <typename IT>
		const T & dim_index(const IT * i) const;

	friend class rstd::MVector<DIM + 1, T, INDEX>;
};

/*===========================================================================*/
// Recursion terminating case of recursively defined multi-dimensional vector.

template <typename T, template <int, typename> class INDEX>
class MVector<1, T, INDEX> :
	public RVector<T>
{
	typedef RVector<T> super;
	typedef MVector<1, T, INDEX> self;
	typedef typename super::size_type size_type;

public:
	virtual ~MVector() {}

	MVector() {}
	MVector(size_t n, const T & value) : super(n, value) {}
	explicit MVector(size_type n) : super(n) {}

	MVector(const super & a) : super(a) {}

	template <typename T2>
		MVector(const MVector<1, T2, INDEX> & a) : super(a) {}

	template <typename InIter>
		MVector(InIter first, InIter last) : super(first, last) {}

// New methods.
	template <typename IT>
		T & operator[](const INDEX<1, IT> & a);

	template <typename IT>
		const T & operator[](const INDEX<1, IT> & a) const;

// Forwarded methods.
	T & operator[](size_type a) { return super::operator[](a); }
	const T & operator[](size_type a) const { return super::operator[](a); }

private:
	template <typename IT>
		const T & dim_index(const IT * i) const { return (*this)[*i]; }

	template <typename IT>
		T & dim_index(const IT * i) { return (*this)[*i]; }

	friend class rstd::MVector<2, T, INDEX>;
};

/*===========================================================================*/
void RVector_test();

/*###########################################################################*/
#include "RVector.cct"
}

#endif
