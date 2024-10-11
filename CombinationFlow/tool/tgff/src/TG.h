// Copyright 2017 by Robert Dick, David Rhodes, and Keith Vallerio.
// All rights reserved.

#ifndef TG_H_
#define TG_H_

/*###########################################################################*/
#include "RString.h"
#include "RVector.h"
#include "Graph.h"

#include <iosfwd>

/*###########################################################################*/
class TGnode {
public:
	TGnode();
	TGnode(long i);
	TGnode(int i, int t, unsigned ri, unsigned ru);
	int			type;		// if == -1, then << generates
	unsigned	ri_;		// indx of resource it runs in (for pack_sched)
	unsigned	ru_;		// indx of resourceuse rep by it (for pack_sched)
	long			name;
	double		deadline;
	bool		hard;
	int			h, w;
	rstd::RVector<double> attrib_;
	rstd::RVector<long> series_children_;
	long series_parent_;
};

/*===========================================================================*/
class TGarc {
public:
	int name;
	int	type;		// if == -1, then << generates
	// constructors
	TGarc(void)  : name(-1), type(-1) 	{}; //	{name = -1; type = -1;};
	TGarc(int n) : name(n), type(-1)	{};	//	{name = n; type = -1;};
	TGarc(int n, unsigned t) : name(n), type(t)	{};	// {name = n; type = t;};
};

/*===========================================================================*/
class TG : public rstd::Graph<TGnode,TGarc> {
public:
// only constructor
	TG();

// augment with (built in) pattern
	void init(int low_bound, int in_deg_req, int out_deg_req, int number,
	  int first_task);

	void init(int number, int in_deg_req, int out_deg_req, double period_req);

	void print_to(std::ostream & os) const;
	void print_to_vcg(std::ostream & os) const;

	friend void	i_PS_dag (std::ostream&, rstd::RVector<TG> &dag, int);

	void push_period();
	double period() const {return period_;}
	long aperiod() const {return aperiod_;}
	int in_deg() const {return in_deg_;}
	int out_deg() const {return out_deg_;}
	void set_period(double p) {period_ = p;}
	void set_aperiod(long ap) {aperiod_ = ap;}

	int max_w() const {return maxw;}
	int max_h() const {return maxh;}

	void compute_dims();

private:
	void augment();
	void add_start_nodes();

	void generate_series_parallel();
	void series_parallel_generate_xover(int local_xover, int global_xover);
	rstd::RVector<long> series_parallel_get_all_parents ();
	bool series_parallel_xarc_added (long a, long b);


	void deadlines();
	int parents_h(long n);

// dag max width and height
		int maxw;
		int maxh;

		int low_bound_;
		int in_deg_;
		int out_deg_;
		double period_;
		long aperiod_;
		int number_;
		int first_task_;
		rstd::RVector<std::pair<long,long> > series_arclist_;
		long num_start_nodes_;
};

/*###########################################################################*/
#endif





