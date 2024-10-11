// Copyright 2017 by Robert Dick.
// All rights reserved.

#ifndef INTERFACE_H_
#define INTERFACE_H_

/*###########################################################################*/
#include "RStd.h"
#include "RFunctional.h"

#include <iosfwd>
#include <iterator>
#include <utility>

/* A better language would provide the means to do the following things
elegantly.  A lesser language wouldn't provide any way to do them at all. */

/* Provide a standard interface for foundation objects.  Although I'm using
inheritance to specify interface, I'm not using the SmallTalk dynamic typing
approach.  Note that much of the complexity below exists to allow the
interface to apply to virtual and non-virtual functions.  Care has been taken
to allow a templatized class designer to define default comparison and
printing methods which are non-virtual (to prevent instantiation) but later
add virtual printing to the class hierarchy. */

/* If you already provided a useful function, e.g., print_to, then don't ask
the interface for one.  If only a default function was provided, then ask the
interface for a useful virtual function by setting PROVIDE to true. */

/* This method of providing interfaces has the potential to change the size of
an object depending on whether or not its contents are printable, etc.  This,
in itself, is O.K.  However, it makes it extremely dangerous to separate
operators for printing an object from that object's definition.  This could
potentially result in different compile-time sizes for the same object
depending on the header files seen to date. */

/* There seems to be a problem with gcc 2.95.2.  Dynamic casts up the
inheritance hierarchy to base class whose first base class (using multiple
inheritance) is non-virtual, results in a clean compiles but severe run-time
errors.  For now, put Prints<> and Comps<> first. */

/*###########################################################################*/
/* Detects objects with an 'ostream & << const T &' operator for their
immediate type and those which are explicitly printable (inherit from Prints<>
at some stage of the class DAG.  Allows printing of such objects.
 If you don't request that it provide a function, make your own print_to,
which may be non-virtual.  Otherwise, make a print_to_default function, which
will be turned into a virtual print_to function only if T is printable. */

namespace rstd {
class PrintsRoot { protected: ~PrintsRoot() {} };

template <typename T, bool PRINTS, bool PROVIDE> class PrintsBase {};

template <typename T>
class PrintsBase<T, true, false> : public virtual PrintsRoot {
	protected: ~PrintsBase() {}
};

template <typename T>
class PrintsBase<T, true, true> : public virtual PrintsRoot {
	public: virtual void print_to(std::ostream & os) const;
	protected: ~PrintsBase() {}
};

template <typename T, bool PROVIDE>
	std::ostream & operator<<(std::ostream & os,
	const PrintsBase<T, true, PROVIDE> & p);

/* Specialize on PROVIDE == false to make re-inheriting interface useful
(once).  More than once will require operator<< to be defined again. */

template <typename T>
	std::ostream & operator<<(std::ostream & os,
	const PrintsBase<T, true, false> & p);

namespace PRINTABLE_NS_ {
	struct NO { char c[65536]; };
	typedef std::ostream & YES;

// Block implicit conversions (motivated by comp.lang.c++.moderated
// discussion)
   struct block {
   	template <typename T>
			block (T const &);
	};

// Default method
	NO operator<<(const block &, const block &);

	template <typename T>
	struct printable {
		enum { result = (same_or_derived<T, PrintsRoot>::result ||
			sizeof(NO) !=
			sizeof(make_instance<std::ostream &>() << make_instance<T>())) };
	};
}

/* Uses print_to function if DEP can be printed.  Provides print_to
function based on print_to_default if PROVIDE is true. */

template <typename T, typename DEP = int, bool PROVIDE = false>
class Prints :
public PrintsBase<T, PRINTABLE_NS_::printable<DEP>::result, PROVIDE> {
	protected: ~Prints() {}
};

template <typename T, typename U,
bool PRINTABLE = PRINTABLE_NS_::printable<T>::result>
class print_if_possible_internal {};

// Using constructors to make it look like a function.
template <typename T, typename U>
struct print_if_possible_internal<T, U, false> {
	static void print(std::ostream & os, const T &, const U & deflt)
		{ os << deflt; }
};

template <typename T, typename U>
struct print_if_possible_internal<T, U, true> {
	static void print(std::ostream & os, const T & t, const U &)
		{ os << t; }
};

template <typename T, typename U>
	void print_if_possible(std::ostream & os, const T & t, const U & deflt)
		{ print_if_possible_internal<T, U>::print(os, t, deflt); }

/*===========================================================================*/
// Prints simple containers.
template <typename InIter>
	void print_cont(InIter first, InIter last, std::ostream & os = std::cout,
	const char * sep = " ");

template <typename Container>
	void print_cont(const Container & c, std::ostream & os, const char * sep = " ");

/*===========================================================================*/
/* LESS will always be less than 0.  EQ will always be 0.  GREATER will always
be greater than 0. */

enum comp_type { LESS = -1, EQ = 0, GREATER = 1 };

std::ostream & operator<<(std::ostream & os, comp_type c);

template <typename T> struct comp_func {
	typedef comp_type (*func)(const T & a, const T & b);
};

class CompsRoot { protected: ~CompsRoot() {} };

template <typename T, bool COMPS, bool PROVIDE> class CompsBase {};

template <typename T>
class CompsBase<T, true, false> : public virtual CompsRoot {
	protected: ~CompsBase() {}
};

template <typename T>
class CompsBase<T, true, true> : public virtual CompsRoot {
	protected: ~CompsBase() {}
	public: virtual comp_type comp(const T & a) const;
};

namespace COMPARABLE_NS_ {
	struct NO { char c[65536]; };
	typedef bool YES;

// Block implicit conversions (motivated by comp.lang.c++.moderated
// discussion)
   struct block {
   	template <typename T>
			block (T const &);
	};

	NO operator<(const block &, const block &);
	NO operator==(const block &, const block &);

	template <typename T>
	struct comparable {
		enum { result = (same_or_derived<T, CompsRoot>::result ||
			((sizeof(NO) != sizeof(make_instance<T>() < make_instance<T>())) &&
			(sizeof(NO) != sizeof(make_instance<T>() == make_instance<T>())))) };
	};
}

template <typename T, typename DEP = int, bool PROVIDE = false>
class Comps :
public CompsBase<T, COMPARABLE_NS_::comparable<DEP>::result, PROVIDE> {
	protected: ~Comps() {}
};

template <typename T>
	comp_type comp(const T & a, const T & b);

// Compare simple containers.
template <typename I,
typename comp_func<typename std::iterator_traits<I>::value_type>::func COMP>
	comp_type comp_cont(I first1, I last1, I first2, I last2);

// Defaults to comp<typename std::iterator_traits<I>::value_type>
template <typename I>
	comp_type comp_cont(I first1, I last1, I first2, I last2);

template <typename C>
	comp_type comp_cont(const C & c1, const C & c2);

template <typename T>
struct comp_obj : public rbinary_function<const T &, const T &, comp_type> {
	comp_type operator()(const T & a, const T & b) const;
};

template <typename T, bool PROVIDE>
	bool operator<(const CompsBase<T, true, PROVIDE> & a,
	const CompsBase<T, true, PROVIDE> & b);

/* Specialize on PROVIDE == false to make re-inheriting interface useful
(once).  More than once will require operator<< to be defined again. */

template <typename T>
	bool operator<(const CompsBase<T, true, false> & a,
	const CompsBase<T, true, false> & b);

template <typename T, bool PROVIDE>
	bool operator==(const CompsBase<T, true, PROVIDE> & a,
	const CompsBase<T, true, PROVIDE> & b);

template <typename T>
	bool operator==(const CompsBase<T, true, false> & a,
	const CompsBase<T, true, false> & b);

template <typename T, bool PROVIDE>
	bool operator!=(const CompsBase<T, true, PROVIDE> & a,
	const CompsBase<T, true, PROVIDE> & b);

template <typename T, bool PROVIDE>
	bool operator!=(const CompsBase<T, true, false> & a,
	const CompsBase<T, true, false> & b);

template <typename T, bool PROVIDE>
	bool operator>(const CompsBase<T, true, PROVIDE> & a,
	const CompsBase<T, true, PROVIDE> & b);

template <typename T, bool PROVIDE>
	bool operator>(const CompsBase<T, true, false> & a,
	const CompsBase<T, true, false> & b);

template <typename T, bool PROVIDE>
	bool operator<=(const CompsBase<T, true, PROVIDE> & a,
	const CompsBase<T, true, PROVIDE> & b);

template <typename T, bool PROVIDE>
	bool operator<=(const CompsBase<T, true, false> & a,
	const CompsBase<T, true, false> & b);

template <typename T, bool PROVIDE>
	bool operator>=(const CompsBase<T, true, PROVIDE> & a,
	const CompsBase<T, true, PROVIDE> & b);

template <typename T, bool PROVIDE>
	bool operator>=(const CompsBase<T, true, false> & a,
	const CompsBase<T, true, false> & b);

/*===========================================================================*/
/* Requires a virtual or non-virtual function of the form 'T * clone() const'.
Check is strict and is made only on clone attempt. */

class ClonesBase { protected: ~ClonesBase() {} };

template <typename> class Clones : public virtual ClonesBase {
	protected: ~Clones() {}
};

/* Smart cloner figures out whether class Clones and uses the class's clone if
possible. */

template <typename T>
	T * clone(const T & a);

/*===========================================================================*/
/* Requires a virtual or non-virtual function of the form 'T * clone() const'.
Check is strict and is made only on rswap attempt. */

class SwapsBase { protected: ~SwapsBase() {} };

template <typename> class Swaps : public virtual SwapsBase {
	protected: ~Swaps() {}
};

/* Smart swapper figures out whether class Swaps and uses the class's rswap if
possible. */

template <typename T>
	void rswap(T & a, T & b);

/*===========================================================================*/
/* Provides functions with the forms 'void self_check() const' and 'void
self_check_deep() const'.  self_check() runs available self-tests on highest
level.  self_check_deep() runs self-tests on contents, as well. */

class SChecksBase { protected: ~SChecksBase() {} };

template <typename T>
class SChecks : public virtual SChecksBase {
	protected: ~SChecks();
};

// Try a deep self-check as long as
template <typename T>
	void try_scheck_deep(const T & a);

template <typename ITER>
	void map_self_check_deep(ITER begin, ITER end);

void Interface_test();

/*###########################################################################*/
#include "Interface.cct"
}
#endif
