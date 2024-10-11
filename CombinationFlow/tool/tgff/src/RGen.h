// Copyright 2017 by Robert Dick.
// All rights reserved.

#ifndef R_GEN_H_
#define R_GEN_H_

/*###########################################################################*/
#include "HolderPtr.h"
#include "Interface.h"
#include "RVector.h"

#include <iosfwd>

namespace rstd {
/*===========================================================================*/
// Random number generator.  True Gaussian pseudo-random variables.

class RGen :
	public Clones<RGen>,
	public Swaps<RGen>
{
	typedef RGen self;

public:
// Construction
	virtual ~RGen();
	explicit RGen(int seed = 1);

// Interface

	virtual self * clone() const { return new self(*this); }
	virtual void rswap(self & a);

// Final
	void set_seed(int seed = 1);
	void set_seed(int i, int j, int k, int l);

// [0, 1)
	double flat01();

// mean = 0, variance = 1
	double gauss01();

// Makes it work with STL random_shuffle().
	int operator()(int ceiling);

// [low, high)
	double flat_range_d(double low, double high);
	long flat_range_l(long low, long high);

	double gauss_mean_d(double mean, double variance);
	long gauss_mean_l(long mean, double variance);

// Only up (if variance is positive) or down (if variance is negative).
	double gauss_mean_half_d(double mean, double variance);
	long gauss_mean_half_l(long mean, double variance);

// Returns 1 if a randomly generated number [0:1) is below the cut-off.
	bool flip(double cut_off = 0.5);

// Get access to a global generator which is allocated on first use.
	static RGen & gen();

private:
	static void gen_test_data(std::ostream & os);

		int ip_, jp_;
		float c_, cd_, cm_;
		HolderPtr<RVector<float> > u_;

	friend void RGen_test();
};

void RGen_test();

/*###########################################################################*/
}
#endif
