// Copyright 2017 by Robert Dick.
// All rights reserved.

#ifndef R_STD_H_
#define R_STD_H_

#include <iostream>
#include <algorithm>

namespace rstd {
// Misc. debugging, type conversion, and control utilities.

// Define ROB_DEBUG to turn on debugging.

/*###########################################################################*/
extern std::ostream cerrhex;
extern std::ostream couthex;

#undef PI
const double PI = 3.14159265358979323846;

// Use for a sane and fast vector of bools.
typedef int big_bool;

/*===========================================================================*/
// Utility

/* In general, macros are dangerous.  I consider the following MAP macros
justified.  This case occurs frequently in algorithms.  These make code easier
to read and maintain. */

#define MAP(a, end_a) \
	for (long a = 0; (a) < (end_a); (a)++)

#define MAP2(a, end_a, b, end_b) \
	for (long a = 0; (a) < (end_a); (a)++) \
		for (long b = 0; (b) < (end_b); (b)++)

#define MAP3(a, end_a, b, end_b, c, end_c) \
	for (long a = 0; (a) < (end_a); (a)++) \
		for (long b = 0; (b) < (end_b); (b)++) \
			for (long c = 0; (c) < (end_c); (c)++)

#define MAP4(a, end_a, b, end_b, c, end_c, d, end_d) \
	for (long a = 0; (a) < (end_a); (a)++) \
		for (long b = 0; (b) < (end_b); (b)++) \
			for (long c = 0; (c) < (end_c); (c)++) \
				for (long d = 0; (d) < (end_d); (d)++)

#define MAPI(a, end_a) \
	for (int a = 0; (a) < (end_a); (a)++)

#define MAPI2(a, end_a, b, end_b) \
	for (int a = 0; (a) < (end_a); (a)++) \
		for (int b = 0; (b) < (end_b); (b)++)

#define MAPI3(a, end_a, b, end_b, c, end_c) \
	for (int a = 0; (a) < (end_a); (a)++) \
		for (int b = 0; (b) < (end_b); (b)++) \
			for (int c = 0; (c) < (end_c); (c)++)

#define MAPI4(a, end_a, b, end_b, c, end_c, d, end_d) \
	for (int a = 0; (a) < (end_a); (a)++) \
		for (int b = 0; (b) < (end_b); (b)++) \
			for (int c = 0; (c) < (end_c); (c)++) \
				for (int d = 0; (d) < (end_d); (d)++)

#define MAPU(a, end_a) \
	for (unsigned long a = 0; (a) < (end_a); (a)++)

#define MAPU2(a, end_a, b, end_b) \
	for (unsigned long a = 0; (a) < (end_a); (a)++) \
		for (unsigned long b = 0; (b) < (end_b); (b)++)

#define MAPU3(a, end_a, b, end_b, c, end_c) \
	for (unsigned long a = 0; (a) < (end_a); (a)++) \
		for (unsigned long b = 0; (b) < (end_b); (b)++) \
			for (unsigned long c = 0; (c) < (end_c); (c)++)

#define MAPU4(a, end_a, b, end_b, c, end_c, d, end_d) \
	for (unsigned long a = 0; (a) < (end_a); (a)++) \
		for (unsigned long b = 0; (b) < (end_b); (b)++) \
			for (unsigned long c = 0; (c) < (end_c); (c)++) \
				for (unsigned long d = 0; (d) < (end_d); (d)++)

/*===========================================================================*/
// Debugging

// Macros need to be used to accomplish this.  Inlines won't work.

// Print position in code.
void rpos(const char * file, int line);
#define Rpos() rstd::rpos(__FILE__, __LINE__)

// Print position in code and abort after printing message.
void rabort(const char * file, int line,
	const char * message = "Program aborted.\n");

void rexit(const char * message = "Exiting.\n");

#define Rabort() rstd::rabort(__FILE__, __LINE__)

// Abort, printing position in code, if (! x).
void rassert(bool x, const char * file, int line);
#define Rassert(x) rstd::rassert((x), __FILE__, __LINE__)

inline void rconfirm(bool x, const char * message) { if (! x) rexit(message); }

// Print variable name and contents.  For temporary debugging use only.
#define Rdump(x) (std::cerr << #x << ": " << (x) << std::endl)
#define Rdumphex(x) (rstd::cerrhex << #x << ": 0x" << (x) << std::endl)

#if 0
// This was needed before static_assert became standard.
template <bool> struct static_assert;
template<> struct static_assert<true> { static_assert() {} };
// No false definition.  Compile-time failure if assertion fails.
#define STATIC_ASSERT(x) rstd::static_assert<(x)>()
#endif

template <int I>
struct int_to_type {
	enum { value = I };
};

template <typename T>
struct type_to_type {
	typedef T orig_type;
};

template <typename T>
struct conversion_exists_helper {
	struct YES { char a; };
	struct NO { char a[5]; };

	static YES test(T);
	static NO test(...);
};

template <typename T>
	T make_instance();

template <typename D, typename B>
struct conversion_exists {
	enum { result =
		(sizeof(conversion_exists_helper<B>::test(
		make_instance<D>())) ==
		sizeof(typename conversion_exists_helper<B>::YES)) };
};

template <typename A, typename B>
struct same_type {
	enum { result = 0 };
};

template <typename A>
struct same_type<A, A> {
	enum { result = 1 };
};

template <typename D, typename B>
struct same_or_derived {
	enum { result = conversion_exists<const D *, const B *>::result &&
		! same_type<const void *, const B *>::result };
};

#define SAME_OR_DERIVED(D, B) \
	(CONVERSION_EXISTS(const D *, const B *) && \
	! SAME_TYPE(const void *, const B *))

#ifdef ROB_DEBUG

#define RDEBUG(x) (x)
#define RASSERT(x) rstd::rassert((x), __FILE__, __LINE__)
#define RCONFIRM(x, y) rstd::rconfirm((x), (y))
#else

#define RDEBUG(x) do {} while (0)
#define RASSERT(x) do {} while (0)
#define RCONFIRM(x, y) do {} while (0)
#endif

/*===========================================================================*/
// The following two are safe.
template <typename T>
	const T & rtoconst(T & x) { return x; }

template <typename T>
	const T * ptoconst(T * x) { return x; }

/* These are potentially dangerous.  Use to avoid duplicate code (and errors)
in const and non-const methods. */

template <typename T>
	T & runconst(const T & x) { return const_cast<T & >(x); }

template <typename T>
	T * punconst(const T * x) { return const_cast<T *>(x); }

// Type conversions which are run-time checked when debugging is on.
int to_signed(unsigned x);
long to_signed(unsigned long x);
unsigned to_unsigned(int x);
unsigned long to_unsigned(long x);

/*===========================================================================*/
void RStd_test();

/*###########################################################################*/
#include "RStd.cct"
}

#endif
