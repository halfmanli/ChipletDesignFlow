// Copyright 2017 by Robert Dick, David Rhodes, and Keith Vallerio.
// All rights reserved.

#include <cmath>
#include <iostream>
#include <algorithm>
#include <set>

#include "RMath.h"
#include "TGraph.h"
#include "ArgPack.h"
#include "RGen.h"
#include "ArgPack.h"
#include "RAlgo.h"

using namespace std;
using namespace rstd;

/*###########################################################################*/
TGraph::TGraph(long tg_offset) :
	dag_(ArgPack::ap().tg_cnt),
	tg_offset_(tg_offset),
	h_period_(-1.0)
{

	if (! dag_.size()) return;

// For use when task names must be unique.
	int first_task = 0;

/* Generate an array of multiples by randomly picking out
of the period_mul_ list. */
	RVector<double> per_mul(dag_.size());
	MAP(x, per_mul.size()) {
		long size = ArgPack::ap().period_mul.size();
		long indx = RGen::gen().flat_range_l(0, size);
		per_mul[x] = ArgPack::ap().period_mul[indx];
	}

// Determine the number of vertices in the graph.
	MAPI (x, dag_.size()) {
		double low_bound =
			RGen::gen().flat_range_d(ArgPack::ap().vertex_cnt_av -
			ArgPack::ap().vertex_cnt_mul,
			ArgPack::ap().vertex_cnt_av + ArgPack::ap().vertex_cnt_mul);

// FIXME: MAKE MORE SOPHISTICATED.
		low_bound *= per_mul[x];

		const int lbnd = static_cast<int>(rint(low_bound));
		dag_[x].init(lbnd, ArgPack::ap().vertex_in_deg,
		  ArgPack::ap().vertex_out_deg, x, first_task);

		first_task += static_cast<int>(dag_[x].size_vertex());
	}

	double deadline_grain = 0.0;

	MAP(x, dag_.size()) {
// Find the average deadline / period_mul for all graphs.
		deadline_grain += dag_[x].max_h() * ArgPack::ap().vertex_time /
			per_mul[x];
	}

	deadline_grain = deadline_grain / static_cast<int>(dag_.size()) * ArgPack::ap().p_laxity;

// Figure out the periods.
//	double deadline_grain =
//		max_deadline / max_multiple * ArgPack::ap().p_laxity;

//cout << "max_deadline: " << max_deadline << ", max_multiple: " <<
//  max_multiple << ", p_laxity_: " << ArgPack::ap().p_laxity_ << "\n";
//cout << "deadline_grain: " << deadline_grain << "\n";

// Round to the nearest vertex_time_.
//	deadline_grain = interval_round(deadline_grain, ArgPack::ap().vertex_time);

//cout << "rounded deadline_grain: " << deadline_grain << "\n";

/* Associate the increasing multipliers with graphs in order of
increasing deadlines. */
	MAP(x, per_mul.size()) {
		const bool is_periodic = RGen::gen().flip(ArgPack::ap().prob_periodic);

		if (is_periodic && ArgPack::ap().period_mul.size()) {
			const double per = per_mul[x] * deadline_grain;
			dag_[x].set_period(per);
		} else {
			dag_[x].set_period(-1.0);
			per_mul[x] = -1;

			long aper_av = ArgPack::ap().aperiodic_min_av;
			long aper_mul = ArgPack::ap().aperiodic_min_mul;
			long aper = aper_av + 
			    RGen::gen().flat_range_l (-aper_mul, aper_mul);
			dag_[x].set_aperiod(aper);
		}
	}

// Ensure that periods >= deadlines if requested.
	if (ArgPack::ap().p_greater_deadline) {
		double new_grain = deadline_grain;

		MAP(x, dag_.size()) {
			double old_period = dag_[x].period();
			dag_[x].push_period();

			double period_dif = dag_[x].period() - old_period;
			if (period_dif > 0.0) {
				new_grain =
				  max(new_grain, deadline_grain + period_dif / per_mul[x]);
			}
		}

		deadline_grain = new_grain;
		MAP(x, dag_.size()) {
			if (dag_[x].period() > 0.0) {
				dag_[x].set_period(per_mul[x] * deadline_grain);
			}
		}
	}

// Compute the hyperperiod
	RVector<double> periodic_per;
	MAP(x, per_mul.size()) {
		if (per_mul[x] > 0.0) {
			periodic_per.push_back(per_mul[x]);
		}
	}

	if (periodic_per.empty()) {
		h_period_ = -1.0;
	} else {
		h_period_ = deadline_grain * lcm(periodic_per);
	}

//cout << "h_period_: " << h_period_ << "\n";

// Count the total number of task types.
// Correct the ArgPack if necessary.

	if (ArgPack::ap().task_unique) {
		ArgPack::write_ap().task_type_cnt = 0;

		MAP(x, dag_.size()) {
			ArgPack::write_ap().task_type_cnt += static_cast<int>(dag_[x].size_vertex());
		}
	}
}

/*===========================================================================*/
void TGraph::print_to(ostream & os) const {
	MAP(x, dag_.size()) {
		os << "@" << ArgPack::ap().tg_label << " " <<
		  (x + tg_offset_) << " {\n";

		dag_[x].print_to(os);
	}

	os << "\n\n";
}

/*===========================================================================*/
void TGraph::print_to_vcg(ostream & os) const {
	MAP(x, dag_.size()) {
		dag_[x].print_to_vcg(os);
	}

	os << "\n\n";
}
