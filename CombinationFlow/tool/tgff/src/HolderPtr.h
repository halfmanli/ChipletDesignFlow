// Copyright 2017 by Robert Dick.
// All rights reserved.

#ifndef HOLDER_PTR_
#define HOLDER_PTR_

/*###########################################################################*/
#include "Interface.h"

#include <algorithm>
#include <typeinfo>

namespace rstd {
template <typename> struct HolderPtrTraits;

/*===========================================================================*/
/* Smart pointer that copies data instead of automatically releasing it when
assigned from. */

template <typename T>
class HolderPtr :
	public Comps<HolderPtr<T> >,
	public Clones<HolderPtr<T> >,
	public Swaps<HolderPtr<T> >
{
	typedef HolderPtr self;

public:
	typedef T element_type;

// Never throws.
	explicit HolderPtr(T * p = 0);

	HolderPtr(const self & a);

	template <typename T2>
		HolderPtr(const HolderPtr<T2> & a);

	self & operator=(const self & a);

	template <class T2>
		self & operator=(const HolderPtr<T2> & a);

// Throws only if destructor of object pointed to throws.
	~HolderPtr();

// Interface
	void rswap(self & hp) throw () { rstd::rswap(data_, hp.data_); }
	self * clone() const { return new self(*this); }
	comp_type comp(const self & a) const;

	T & operator*() const throw () { return *data_; }
	T * operator->() const throw () { return data_; }
	T * get() const throw () { return data_; }

// Never throws.
	T * release();
// Never throws.
	void reset(T * p = 0);

private:
		T * data_;
};

/*###########################################################################*/
#include "HolderPtr.cct"
}
#endif
