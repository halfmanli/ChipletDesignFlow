// Copyright 2017 by Robert Dick.
// All rights reserved.

#ifndef R_FUNCTIONAL_H_
#define R_FUNCTIONAL_H_

/* Defines functional object interfaces similar to STL but with protection
against deletes through a pointer to the base class.  Provides simple
functional objects. */

namespace rstd {
/*###########################################################################*/
template <class Arg, class Result>
struct runary_function {
	typedef Arg argument_type;
	typedef Result result_type;

protected:
	~runary_function() {}
};

/*===========================================================================*/
template <typename Arg1, typename Arg2, typename Result>
struct rbinary_function {
	typedef Arg1 first_argument_type;
	typedef Arg2 second_argument_type;
	typedef Result result_type;

protected:
	~rbinary_function() {}
};

/*###########################################################################*/
template <typename T>
	struct ptr_dereference : public runary_function<T *, T &>
	{ T & operator()(T * x) { return *x; } };

template <typename T>
	struct ptr_reference : public runary_function<T &, T *>
	{ T * operator()(T & x) { return &x; } };

/*===========================================================================*/
template <typename T>
struct deref_less : public rbinary_function<const T *, const T *, bool> {
	bool operator()(const T * a, const T * b) { return *a < *b; }
};

/*===========================================================================*/
template <typename T>
struct deref_equal_to : public rbinary_function<const T *, const T *, bool> {
	bool operator()(const T * a, const T * b) { return *a == *b; }
};

}
#endif
