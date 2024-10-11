// Copyright 2017 by Robert Dick, David Rhodes, and Keith Vallerio.
// All rights reserved.

#ifndef P_GRAPH_H_
#define P_GRAPH_H_

/*###########################################################################*/
#include "RVector.h"
#include "TG.h"

#include <string>
#include <iosfwd>

class ResourceInstance;
class Resource;

// --------------------------------------------------------------------------
class ResourceAttrib {
	friend class Resource;
	friend class PGraph;
	friend double com_possible(std::string aname, double t1, double t2,
							   rstd::RVector<ResourceInstance> &coms,
							   rstd::RVector<Resource> &com_types,
							   double arc_fill_factor);
private:
	ResourceAttrib	(std::string nm, double v) : name_(nm), value_(v) {};
	std::string  name_;
	double value_;
};

class ResourceUse {
	friend class ResourceInstance;
	friend class PGraph;
	friend ResourceUse *NextToStart(double t, rstd::RVector<ResourceUse>& ruv);
	friend double com_possible(std::string aname, double t1, double t2,
							   rstd::RVector<ResourceInstance> &coms,
							   rstd::RVector<Resource> &com_types,
							   double arc_fill_factor);
private:
	ResourceUse	(std::string nm, double start, double end)
					: name_(nm), ts_(start), te_(end) {};
	std::string	name_;
	double	ts_, te_;
	void print_to(std::ostream &os) const;
	friend std::ostream &operator<<(std::ostream &os, /**/const ResourceUse &ru)
	  {ru.print_to(os); return os;}
};

class Resource {
	friend class ResourceInstance;
	friend class PGraph;
private:
	std::string					name_;
	rstd::RVector<ResourceAttrib>	attrib_;
	rstd::RVector<double>			exec_time_;	// for PE resources only

	void print_to(std::ostream &os) const;
	friend std::ostream &operator<<(std::ostream &os, /**/ const Resource &r)
	  {r.print_to(os); return os;}
public:
	Resource();
	ResourceAttrib& GetAttrib(std::string s);
	void SetAttrib(std::string s, double v);
};

class ResourceInstance {
	friend class PGraph;
	friend double com_possible(std::string aname, double t1, double t2,
							   rstd::RVector<ResourceInstance> &coms,
							   rstd::RVector<Resource> &com_types,
							   double arc_fill_factor);
private:
	ResourceInstance(std::string nm, unsigned r, double tf)
					: name_(nm), resource_indx_(r), time_frame_(tf), ru_() {};

	std::string					name_;
	unsigned				resource_indx_;
	double					time_frame_;
	rstd::RVector<ResourceUse>	ru_;

	void print_to(std::ostream &os) const;
	friend std::ostream &operator<<(std::ostream &os,const ResourceInstance &ri)
	  {ri.print_to(os); return os;}
};

/*###########################################################################*/
class PGraph {
public:
	PGraph(	std::ostream &eps_out,
			int	   num_task_graphs,
			int	   avg_tasks_per_pe,
			double avg_task_time,	/* PE STUFF */
			double mul_task_time,
			double task_slack,
			double task_round,
			int    num_pe_types,
			int    num_pe_soln,
			int    num_com_types,	/* COM STUFF */
			int    num_com_soln,
			double arc_fill_factor);
	void print_to(std::ostream &os) const;
	void print_to_ps(std::ostream &os, const std::string & filename) const;
	double h_period() const {return h_period_;}

private:
	rstd::RVector<Resource>			pe_;
	rstd::RVector<ResourceInstance>	pe_soln_;
	rstd::RVector<Resource>			com_;
	rstd::RVector<ResourceInstance>	com_soln_;
	rstd::RVector<int>				arc_data_size_;
	rstd::RVector<TG>				dag_; 
	double						cost_of_soln_;
	double h_period_;		// only non-periodic now, keep for future
};

/*###########################################################################*/
#endif

