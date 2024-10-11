// Copyright 2017 by Robert Dick, David Rhodes, and Keith Vallerio.
// All rights reserved.

// Warning: This code uses 'new' in a non-exception-safe manner
//			(e.g. causing core dumps in low memory conditions).
//			However, this is suitable for a non-memory intensive
//			application.
//

#include <cmath>
#include <iostream>

#include "PGraph.h"
#include "ArgPack.h"
#include "RGen.h"
#include "RMath.h"
#include "float.h"
#include "RStd.h"

using namespace std;
using namespace rstd;

// --------------------------------------------------
void ResourceUse::print_to(ostream &os) const
{
	os << name_ << "(s/e = " << ts_ << "/" << te_ << ")";
}

// --------------------------------------------------
Resource::Resource() :
	name_(),
	attrib_(),
	exec_time_()
{}

// --------------------------------------------------
void Resource::print_to(ostream &os) const
{
	os << endl << "@" << name_ << " {" << endl << "#";

	MAP(x,attrib_.size())			// write attributes
		os << "\t" << attrib_[x].name_ ;
	os << endl;
	MAP(x,attrib_.size())
		os << "\t" << attrib_[x].value_ ;
	os << endl;

	// write type dependent lists
	os << endl << "#\ttype\texec_time" << endl;
	MAP(x,exec_time_.size())
		os << "\t" << x << "\t" << exec_time_[x] << endl;

	os << "}" << endl;	// complete
}
	
// --------------------------------------------------
void ResourceInstance::print_to(ostream &os) const
{
	os	<< "# Instance Name " << name_
		<< "  \t--resource_index= " << resource_indx_ << endl;
	double util = 0.0;
	MAP(x,ru_.size())
	{
		os	<< "#  " << ru_[x].name_ << " \tstart/end:\t"
			<< ru_[x].ts_ << " / " << ru_[x].te_ << endl;
		util += (ru_[x].te_ - ru_[x].ts_) / time_frame_;
	}
	os << "# Utilization = " << (100.0*util) << "%" << endl;

	os << endl;
}
// --------------------------------------------------
	 
// UTILITY DEFS
#define TTSTART(n) ((pe_soln_[(n)->ri_].ru_[(n)->ru_]).ts_)
#define TTEND(n)   ((pe_soln_[(n)->ri_].ru_[(n)->ru_]).te_)

// --------------------------------------------------
static inline
bool in_range(double v, double range_low, double range_hgh)
{
	return (v >= range_low && v <= range_hgh);
}
// --------------------------------------------------
ResourceUse *NextToStart(double t, RVector<ResourceUse>& ruv)
{
	ResourceUse	*n = 0;
	MAP(x,ruv.size())
		if( ruv[x].ts_ >= t && (!n || n->ts_ > ruv[x].ts_) )
			n = &ruv[x];

	return n;
}
// --------------------------------------------------
class Dpair {
public:
	Dpair(double s, double room) : start_at(s), space(room) {};
	double start_at, space;
};
// --------------------------------------------------
double com_possible(string aname, double t1, double t2,
					RVector<ResourceInstance> &coms,
					RVector<Resource> &com_types,
					double arc_fill_factor)
{
	// since non-preemptive com is being used, look for largest open
	// space within [t1,t2] ... find the largest "arc data size" that
	// is possible ... returns the arc data size (0 if not possible)

	RASSERT( t2 >= t1 );

	RVector<Dpair>	dpair( coms.size(), Dpair(0.0,0.0) );

	MAP(x,coms.size())
	{	
		// find max open space on resource within [t1,t2].
		// nasty O(n^2) routine ... could be O(n log n) if we sorted

		RVector<ResourceUse> &ruv= coms[x].ru_;
		ResourceUse *ru = NextToStart(0.0, ruv);
		if( !ru )
		{
			dpair[x] = Dpair( t1, t2-t1 ); // all the space you want
		}
		else
		{
			while( ru )
			{
				ResourceUse *follow = NextToStart( ru->te_, ruv);
				double t3 = ru->te_;
 				double t4 = ( follow ? follow->ts_ : FLT_MAX);
				RASSERT( t4 >= t3 );
				
				if     ( t3 <= t1 && t4 >= t2 )
					dpair[x] = Dpair( t1, t2-t1 );	// all the space you want
				else if( t3 >= t1 && t4 <= t2 )			// inside
				{
					if( dpair[x].space < (t4-t3) )
						dpair[x] = Dpair(t3, t4-t3 );	// better
				}
				else if( in_range(t3,t1,t2) )
				{
					if( dpair[x].space < (t2-t3) )
						dpair[x] = Dpair(t3, t2-t3 );	// better
				}
				else if( in_range(t4,t1,t2) )
				{
					if( dpair[x].space < (t4-t1) )
						dpair[x] = Dpair(t1, t4-t1 );	// better
				}

				ru = follow;
			}
		}
	}

	// find com which supports largest COM data on ARC (if any)

	int		selct = -1;
	double	max_arc_data_size = 0.0;
	MAPI (x, dpair.size())
	{
		Resource &com_type = com_types[coms[x].resource_indx_];
		ResourceAttrib &ra_ds = com_type.GetAttrib("datarate");
		double arc_data_size = dpair[x].space * ra_ds.value_;
		if( arc_data_size > max_arc_data_size )
		{
			max_arc_data_size = arc_data_size;
			selct = x;
		}
	}

	if( selct >= 0)		// YEAH, we have one!
	{
		double space = arc_fill_factor * dpair[selct].space;
		if( space < 1.0 )
		{	// arc must be at least 1 com unitsize
			return 0.0;
		}
		double end = dpair[selct].start_at + space;
		ResourceUse ru = ResourceUse(aname, dpair[selct].start_at, end);
		coms[selct].ru_.push_back(ru);
	}

	return max_arc_data_size;
}
// --------------------------------------------------
static
bool out_arc_ok(TG& dag, long node)
{
	if( dag.out_deg() >= 0 )
		return (dag.vertex(node)->size_out() < dag.out_deg());

	return true;	// no limit
}
// --------------------------------------------------
static
bool in_arc_ok(TG& dag, long node)
{
	if( dag.in_deg() >= 0 )
		return (dag.vertex(node)->size_in() < dag.in_deg());

	return true;	// no limit
}

/*###########################################################################*/
PGraph::PGraph( ostream &eps_out,
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
				double arc_fill_factor)
	:	pe_(num_pe_types),
		pe_soln_(),
		com_(num_com_types),
		com_soln_(),
		arc_data_size_(0),
		dag_(num_task_graphs),
		cost_of_soln_(0.0),
		h_period_(0.0)
{
	RGen::gen().set_seed(ArgPack::ap().seed);

	// create PE types
	MAP(x,pe_.size())
	{
		pe_[x].name_ = string("PE ") + to_string(x);
		double cost = RGen::gen().flat_range_d( 50.0, 150.0);
		pe_[x].attrib_.push_back(ResourceAttrib("cost",cost));
		pe_[x].attrib_.push_back(ResourceAttrib("n_interrupts",0));
		pe_[x].attrib_.push_back(ResourceAttrib("interrupt_time",0));
	}

	// create COM types
	MAP(x,com_.size())
	{
		com_[x].name_ = string("COM ") + to_string(x);
		double cost = RGen::gen().flat_range_d( 50.0, 150.0);
		com_[x].attrib_.push_back(ResourceAttrib("cost",cost));
		com_[x].attrib_.push_back(ResourceAttrib("n_interrupts",0));
		com_[x].attrib_.push_back(ResourceAttrib("interrupt_time",0));
		com_[x].attrib_.push_back(ResourceAttrib("datarate",132));
		com_[x].attrib_.push_back(ResourceAttrib("nconnects",8));
		com_[x].attrib_.push_back(ResourceAttrib("code",2));
	}

	// figure PERIOD -- constant for all task graphs...
	h_period_ = avg_tasks_per_pe * (avg_task_time + task_slack);
	h_period_ = interval_round(h_period_, task_round);

	MAPI (x,dag_.size())
	{
		dag_[x].init(x, ArgPack::ap().vertex_in_deg,
			ArgPack::ap().vertex_out_deg, h_period_);
	}

	// CREATE SOLUTION
	int tot_task_types = 0;

	for(int k=0; k < num_pe_soln; k++ )		// create tasks first
	{
		// pick resource for instance
		unsigned pe_indx = static_cast<int>(RGen::gen().flat_range_l(0, num_pe_types));
		ResourceInstance ri = ResourceInstance(string("PEins-") + to_string(k),
									pe_indx, h_period_);
		pe_soln_.push_back(ri);

		ResourceAttrib& racost = pe_[pe_indx].GetAttrib("cost");
		cost_of_soln_ += racost.value_;

		// pack up with ResourceUse (Tasks here)
		double so_far = 0.0;

		while( so_far < h_period_ )
		{
			double y = RGen::gen().flat_range_d(-1.0,1.0);
			double t = avg_task_time + y * mul_task_time;	// gen task time
			if( task_round != 0.0 )
				t = interval_round(t,task_round);			// round it
		
			t = min( t, h_period_ - so_far );				// clip it

			// randomly assign to a task-graph..
			int wdag = static_cast<int>(RGen::gen().flat_range_l(0, num_task_graphs));

			string	task_name = string("t") + to_string(wdag) + "_" +
				to_string(tot_task_types);
			ResourceUse ru = ResourceUse(task_name,so_far,t+so_far);
			pe_soln_.back().ru_.push_back(ru);

			dag_[wdag].add_vertex (TGnode(tot_task_types,tot_task_types,
									  	  static_cast<int>(pe_soln_.size()) - 1,
									  	  static_cast<int>(pe_soln_.back().ru_.size()) - 1));
			so_far += (t + task_slack);
			tot_task_types++;
		}
	}

	// fix COM resource attribute for datarate
	for(int k=0; k < num_com_types; k++ )
	{
		double dr = RGen::gen().flat_range_d(50.0,150.0);
		com_[k].SetAttrib("datarate",dr);
	}

	// create COM resources for solution
	for(int k=0; k < num_com_soln; k++ )
	{
		// pick resource for instance
		int indx = static_cast<int>(RGen::gen().flat_range_l(0,num_com_types));
		ResourceInstance ci = ResourceInstance(string("COMins-") + to_string(k),
								indx, h_period_);
		com_soln_.push_back(ci);

		ResourceAttrib& racost = com_[indx].GetAttrib("cost");
		cost_of_soln_ += racost.value_;
	}

	// CREATE ARCS ...
	int arc_count = 0;
	MAP(x,dag_.size())
	{
		RVector<long>	dag_nodes;	// make sublist of these nodes
		MAP (y, dag_[x].size_vertex()) {
			dag_nodes.push_back(y);
		}

		MAP (y, dag_nodes.size()) {
			TGnode *n1 = &dag_[x][y];
			double n1end = TTEND(n1);
			MAP (z, dag_nodes.size()) {
				TGnode *n2 = &dag_[x][z];
				if( !(out_arc_ok(dag_[x],y) && in_arc_ok(dag_[x],z)) )
					continue;
				double n2beg = TTSTART(n2);

				if( n2beg > n1end )
				{	// candidate for causal arc
					double arc_size;
					// if there is an arc, this will be its name:
					string arcName = string("a") + to_string(x) + "_" +
						to_string(arc_count);
					if( n1->ri_ == n2->ri_ )
					{	// on same resource, since "no penalty" is assumed
						// could make arc data as large as we want ... making
						// it too big though could give a "clue" to the
						// co-design tools.  OK, so make it 20% of task
						// size assuming COM type 0
						Resource &com_type = com_[0];
						ResourceAttrib &ra_ds = com_type.GetAttrib("datarate");
						arc_size = 0.2 * avg_task_time * ra_ds.value_;

						if( arc_size >= 1.0 )
						{
							arc_data_size_.push_back( int(arc_size) );

							dag_[x].add_edge(y, z, TGarc(arc_count,arc_count));
							arc_count++;
						}
					}
					else if( (arc_size = com_possible(arcName,
													  n1end,
													  n2beg,
													  com_soln_,
													  com_,
													  arc_fill_factor)) > 0.0 )
					{	// com_possible added the resource utilization
						dag_[x].add_edge(y,z,TGarc(arc_count,arc_count));
						arc_count++;
						arc_data_size_.push_back( int(arc_size) );
					}
				}
			}
		}
	}

	// generate EXEC Time tables for pe_types (note: we already know solution
	// and its time)
	MAP (y, dag_.size()) {
		MAP(z, dag_[y].size_vertex()) {
			TGnode *node = &dag_[y][z];
			ResourceInstance&	ri = pe_soln_[ node->ri_ ];
			ResourceUse&		ru = ri.ru_  [ node->ru_ ];
			double	soln_runtime = ru.te_ - ru.ts_;
			soln_runtime = TTEND(node) - TTSTART(node);
			MAP(x,pe_.size())
			{
				if( ri.resource_indx_ == unsigned(x) )
					pe_[x].exec_time_.push_back( soln_runtime );
				else
					pe_[x].exec_time_.push_back( 2.0*soln_runtime );
			}
		}
	}
	
	// CREATE Deadlines ... as in TGFF, put deadlines
	// on those with no children
	MAP(x,dag_.size())
	{
		MAP(y, dag_[x].size_vertex()) {
			if( dag_[x].vertex(y)->size_out() == 0 )
				dag_[x][y].deadline = TTEND(&dag_[x][y]);
				// deadline is tight, use task_slack for general loosening
		}
		dag_[x].compute_dims();
	}

	// wrap up with EPS generation
	print_to_ps(eps_out, ArgPack::ap().eps_file_name);
}

/*===========================================================================*/
void PGraph::print_to(ostream &os) const {
	MAP(x, pe_.size()) {
		os << pe_[x];
	}

	os << "\n# Known solution cost = " << cost_of_soln_ << endl;
	os << "# Here's the known solution with " << pe_soln_.size() << " PEs" << endl;
	MAP(x, pe_soln_.size())
		os << pe_soln_[x];

	MAP(x, com_.size())
		os << com_[x];

	os << "\n# Here's the associated COM solution with "
		<< com_soln_.size() << " COMs" << endl;
	MAP(x, com_soln_.size())
		os << com_soln_[x];

	os << "@ARCDATASIZETABLE 0 {" << endl
		<< "#  type	\tdata_size" << endl;
	MAP(x, arc_data_size_.size()) 
		os << "  " << x << "\t" << arc_data_size_[x] << endl;
	os << "}" << endl;

	MAP(x, dag_.size()) {
		os << "\n@TASK_GRAPH" << " " <<
		  x << " {\n";
		dag_[x].print_to(os);
	}

	os << "\n\n";
}

/*===========================================================================*/
ResourceAttrib& Resource::GetAttrib(string n)
{
	MAP(x,attrib_.size())
		if( attrib_[x].name_ == n ) return attrib_[x];

	Rabort();	// fail
	return attrib_[0];
}
/*===========================================================================*/
void Resource::SetAttrib(string n, double v)
{
	MAP(x,attrib_.size())
		if( attrib_[x].name_ == n ) 
		{
			attrib_[x].value_ = v;
			return;
		}

	Rabort();	// fail
}
