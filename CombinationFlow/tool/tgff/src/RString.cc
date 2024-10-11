// Copyright 2017 by Robert Dick.
// All rights reserved.

#include "RString.h"
#include "RVector.h"
#include "RStd.h"
#include "RMath.h"

#include <algorithm>
#include <functional>
#include <string>
#include <cstring>
#include <iostream>
#include <cmath>

namespace rstd {
using namespace std;

/*###########################################################################*/
const RVector<string>
tokenize(const string & s, const char * delim) {
	RVector<string> result;

	if (s.empty())
		return result;

	string::size_type i = s.find_first_not_of(delim);

	do {
		string::size_type next = s.find_first_of(delim, i);
		result.push_back(string(s, i, next - i));
		next = s.find_first_not_of(delim, next);
		i = next;
	} while (i < s.size());

	return result;
}

/*===========================================================================*/
const string first_token(const string & s, const char * delim) {
	string::size_type start = s.find_first_not_of(delim);
	string::size_type end = s.find_first_of(delim, start);

	if (start >= s.size() || start > end)
		return string();
	else
		return string(s, start, end - start);
}

/*===========================================================================*/
void pop_token(string & s, const char * delim) {
	string::size_type i = s.find_first_not_of(delim);
	i = s.find_first_of(delim, i);
	i = s.find_first_not_of(delim, i);
	s.erase(0, i);	
}

/*===========================================================================*/
bool
insen_less<string>::operator()(const string & a, const string & b) const {
	return lexicographical_compare(a.begin(), a.end(), b.begin(), b.end(),
	  insen_less<char>());
}

/*===========================================================================*/
bool
insen_equal_to<string>::operator()(const string & a, const string & b) const {
	return a.size() == b.size() && equal(a.begin(), a.end(), b.begin(),
	  insen_equal_to<char>());
}

/*###########################################################################*/
namespace {
	template <typename T>
	T convert(const string & s) {
		istringstream iss(s.c_str());
		double dbl;
		iss >> dbl;
		Rassert(dbl <= rnlimits<T>::max());
		Rassert(dbl >= rnlimits<T>::smallest());
		Rassert(! iss.bad());
		return static_cast<T>(dbl);
	}

	template <>
	bool convert<bool>(const string & s) {
		insen_equal_to<string> eq;
		if (eq(s, "true") || eq(s, "t") || eq(s, "1")) {
			return true;
		} else if (eq(s, "false") || eq(s, "f") || eq(s, "0")) {
			return false;
		}

		Rdump(s);
		Rabort();
		return false;
	}
}

/*===========================================================================*/
Conv::Conv(const string & str) :
	str_(str)
{}

/*===========================================================================*/
Conv::operator bool()
	{ return convert<bool>(str_); }

Conv::operator signed char()
	{ return convert<char>(str_); }

Conv::operator unsigned char()
	{ return convert<unsigned char>(str_); }

Conv::operator short()
	{ return convert<short>(str_); }

Conv::operator unsigned short()
	{ return convert<unsigned short>(str_); }

Conv::operator int()
	{ return convert<int>(str_); }

Conv::operator unsigned()
	{ return convert<unsigned>(str_); }

Conv::operator long()
	{ return convert<long>(str_); }

Conv::operator unsigned long()
	{ return convert<unsigned long>(str_); }

Conv::operator float()
	{ return convert<float>(str_); }

Conv::operator double()
	{ return convert<double>(str_); }

/*===========================================================================*/
void RString_test() {
	const char * str = "This is a string    waiting    to be tokenized.\n";
	cout << str;
	tokenize(str).print_to_default(cout);
	cout << "\n";
	unsigned k = Conv("53");
	Rassert(k == 53);
	double l = Conv("123401.23");
	Rassert(fabs(l - 123401.23) < 0.0000001);

	string fred = "This is a string waiting for its first token to be "
		"popped.\n";

	string ft = first_token(fred);
	pop_token(fred);
	cout << ft << "***" << fred;

	string gene = "This_is_a_string_without_tokens";
	string emp = "";

	string ft2 = first_token(gene);
	cout << "Should be whole string: " << ft2 << "\n";
	string ft3 = first_token(emp);
	cout << "Should be ****: **" << ft3 << "**\n";

	RVector<string> sv = tokenize(string(""));
}

}
