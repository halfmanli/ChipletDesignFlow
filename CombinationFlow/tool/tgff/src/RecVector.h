// Copyright 2017 by Robert Dick.
// All rights reserved.

#ifndef REC_VECTOR_H_
#define REC_VECTOR_H_

/*###########################################################################*/
#include <iosfwd>
#include <utility>

#ifdef ROB_DEBUG
#	include <typeinfo>
#endif

#include "Interface.h"
#include "RStd.h"
#include "RVector.h"
#include "FVector.h"

namespace rstd {

/*===========================================================================*/
/* Rectangular vector with bounds checking.  No speed hits when compiled with
ROB_DEBUG not defined. */

template <typename T>
class RecVector2 :
	public Clones<RecVector2<T> >,
	public Swaps<RecVector2<T> >,
	public SChecks<RecVector2<T> >
{
	typedef RVector<T> impl;
	typedef RecVector2 self;

public:
// Typedefs
	typedef typename impl::value_type value_type;
	typedef typename impl::pointer pointer;
	typedef typename impl::const_pointer const_pointer;
	typedef typename impl::iterator iterator;
	typedef typename impl::const_iterator const_iterator;
	typedef typename impl::reference reference;
	typedef typename impl::const_reference const_reference;
	typedef long size_type;
	typedef typename impl::difference_type difference_type;
	typedef typename impl::allocator_type allocator_type;
	typedef typename impl::const_reverse_iterator const_reverse_iterator;
	typedef typename impl::reverse_iterator reverse_iterator;

// Construction
	virtual ~RecVector2();

	RecVector2() : impl_(), size_(0) {}
	RecVector2(size_type sx, size_type sy, const T & value);
	RecVector2(size_type sx, size_type sy);
	virtual self & operator=(const self & a);

// Interface
	virtual self * clone() const { return new self(*this); }
	virtual void rswap(self &rv);
	virtual void self_check() const {}
	virtual void self_check_deep() const { impl_.self_check_deep(); }

// Final
	void resize(size_type x, size_type y, const T & val);
	void resize(size_type x, size_type y);

	T & operator()(size_type a, size_type b);
	const T & operator()(size_type a, size_type b) const;

	const FVector<2, size_type> & size() const { return size_; }

	void assign(size_type x_first, size_type y_first, size_type x_last,
		size_type y_last, const self & vec, size_type vec_x_first = 0,
		size_type vec_y_first = 0);

	void clear() { impl_.clear(); size_[0] = 0; size_[1] = 0; }

	iterator begin() { return impl_.begin(); }
	const_iterator begin() const { return impl_.begin(); }
	iterator end() { return impl_.end(); }
	const_iterator end() const { return impl_.end(); }
	iterator rbegin() { return impl_.rbegin(); }
	const_iterator rbegin() const { return impl_.rbegin(); }
	iterator rend() { return impl_.rend(); }
	const_iterator rend() const { return impl_.rend(); }
	bool empty() const { return impl_.empty(); }

private:
		impl impl_;
		FVector<2, size_type> size_;
};

/*===========================================================================*/
template <typename T>
class RecVector3 :
	public Clones<RecVector3<T> >,
	public Swaps<RecVector3<T> >,
	public SChecks<RecVector3<T> >
{
	typedef RVector<T> impl;
	typedef RecVector3 self;

public:
// Typedefs
	typedef typename impl::value_type value_type;
	typedef typename impl::pointer pointer;
	typedef typename impl::const_pointer const_pointer;
	typedef typename impl::iterator iterator;
	typedef typename impl::const_iterator const_iterator;
	typedef typename impl::reference reference;
	typedef typename impl::const_reference const_reference;
	typedef long size_type;
	typedef typename impl::difference_type difference_type;
	typedef typename impl::allocator_type allocator_type;
	typedef typename impl::const_reverse_iterator const_reverse_iterator;
	typedef typename impl::reverse_iterator reverse_iterator;

// Construction
	virtual ~RecVector3() {}

	RecVector3() : size_(0) {}
	RecVector3(size_type sx, size_type sy, size_type sz, const T & value);
	RecVector3(size_type sx, size_type sy, size_type sz);
	virtual self & operator=(const self & a);

// Interface
	virtual self * clone() const { return new self(*this); }
	virtual void rswap(self & rv);
	virtual void self_check() const {}
	virtual void self_check_deep() const { impl_.self_check_deep(); }

// Final
	void resize(size_type x, size_type y, size_type z, const T & val);
	void resize(size_type x, size_type y, size_type z);

	T & operator()(size_type a, size_type b, size_type c);
	const T & operator()(size_type a, size_type b, size_type c) const;

	const FVector<3, size_type> & size() const { return size_; }

	void assign(size_type x_first, size_type y_first, size_type z_first,
		size_type x_last, size_type y_last, size_type z_last, const self & vec,
		size_type vec_x_first = 0, size_type vec_y_first = 0,
		size_type vec_z_first = 0);

	iterator begin() { return impl_.begin(); }
	const_iterator begin() const { return impl_.begin(); }
	iterator end() { return impl_.end(); }
	const_iterator end() const { return impl_.end(); }
	iterator rbegin() { return impl_.rbegin(); }
	const_iterator rbegin() const { return impl_.rbegin(); }
	iterator rend() { return impl_.rend(); }
	const_iterator rend() const { return impl_.rend(); }
	bool empty() const { return impl_.empty(); }

private:
		impl impl_;
		FVector<3, size_type> size_;
};

/*###########################################################################*/
template <typename T>
class RecVector4 :
	public Clones<RecVector4<T> >,
	public Swaps<RecVector4<T> >,
	public SChecks<RecVector4<T> >
{
	typedef RVector<T> impl;
	typedef RecVector4 self;

public:
// Typedefs
	typedef typename impl::value_type value_type;
	typedef typename impl::pointer pointer;
	typedef typename impl::const_pointer const_pointer;
	typedef typename impl::iterator iterator;
	typedef typename impl::const_iterator const_iterator;
	typedef typename impl::reference reference;
	typedef typename impl::const_reference const_reference;
	typedef long size_type;
	typedef typename impl::difference_type difference_type;
	typedef typename impl::allocator_type allocator_type;
	typedef typename impl::const_reverse_iterator const_reverse_iterator;
	typedef typename impl::reverse_iterator reverse_iterator;

// Construction
	virtual ~RecVector4() {}

	RecVector4() : size_(0) {}

	RecVector4(size_type sx, size_type sy, size_type sz, size_type sp,
		const T & value);

	RecVector4(size_type sx, size_type sy, size_type sz, size_type sp);
	virtual self & operator=(const self & a);

// Interface
	virtual self * clone() const { return new self(*this); }
	virtual void rswap(self &rv);
	virtual void self_check() const {}
	virtual void self_check_deep() const { impl_.self_check_deep(); }

// Final
	void resize(size_type x, size_type y, size_type z, size_type p,
		const T & val);

	void resize(size_type x, size_type y, size_type z, size_type p);
	T & operator()(size_type a, size_type b, size_type c, size_type d);

	const T & operator()(size_type a, size_type b, size_type c,
		size_type d) const;

	const FVector<4, size_type> & size() const { return size_; }

	void assign(size_type x_first, size_type y_first, size_type x_last,
		size_type y_last, size_type z_first, size_type z_last,
		size_type p_first, size_type p_last, const self & vec,
		size_type vec_x_first = 0, size_type vec_y_first = 0,
		size_type vec_z_first = 0, size_type vec_p_first = 0);

	iterator begin() { return impl_.begin(); }
	const_iterator begin() const { return impl_.begin(); }
	iterator end() { return impl_.end(); }
	const_iterator end() const { return impl_.end(); }
	iterator rbegin() { return impl_.rbegin(); }
	const_iterator rbegin() const { return impl_.rbegin(); }
	iterator rend() { return impl_.rend(); }
	const_iterator rend() const { return impl_.rend(); }
	bool empty() const { return impl_.empty(); }

private:
		impl impl_;
		FVector<4, size_type> size_;
};

/*===========================================================================*/
void RecVector_test();

/*###########################################################################*/
#include  "RecVector.cct"
}
#endif
