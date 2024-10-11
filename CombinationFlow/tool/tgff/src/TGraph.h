// Copyright 2017 by Robert Dick, David Rhodes, and Keith Vallerio.
// All rights reserved.

#ifndef T_GRAPH_H_
#define T_GRAPH_H_

/*###########################################################################*/
#include "RVector.h"
#include "TG.h"
#include "RString.h"

#include <iosfwd>

/*###########################################################################*/
class TGraph {
public:
	TGraph(long offset);
	void print_to(std::ostream & os) const;
	void print_to_vcg(std::ostream & os) const;
	void print_to_ps(std::ostream & os, const std::string & filename) const;
	double h_period() const { return h_period_; }

private:
		rstd::RVector<TG> dag_; 
		long tg_offset_;
		double h_period_;
};

/*###########################################################################*/
#endif

