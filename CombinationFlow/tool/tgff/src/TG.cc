// Copyright 2017 by Robert Dick, David Rhodes, and Keith Vallerio.
// All rights reserved.

#include "TG.h"

#include <iomanip>
#include <algorithm>
#include <cmath>

#include "RMath.h"
#include "Epsilon.h"
#include "RVector.h"
#include "RStd.h"
#include "RGen.h"
#include "ArgPack.h"

using namespace std;
using namespace rstd;

/*###########################################################################*/
TGnode::TGnode() :
	type(-1),
	ri_(0U),
	ru_(0U),
	name(-1),
	deadline(0.0),
	hard(true),
	h(0),
	w(0),
	attrib_(),
	series_children_(),
	series_parent_(-1)
{}

/*===========================================================================*/
TGnode::TGnode(long i) :
	type(-1),
	ri_(0U),
	ru_(0U),
	name(i),
	deadline(0.0),
	hard(true),
	h(0),
	w(0),
	attrib_(),
	series_children_(),
	series_parent_(-1)
{
	MAP(x, ArgPack::ap().task_attrib_av.size()) {
		double val = ArgPack::ap().task_attrib_av[x] +
		  RGen::gen().flat_range_d(-ArgPack::ap().task_attrib_mul[x],
		  ArgPack::ap().task_attrib_mul[x]);

		if (eps_not_equal_to<double>()(ArgPack::ap().task_attrib_round[x], 0.0)) {
			val = interval_round(val, ArgPack::ap().task_attrib_round[x]);
		}

		attrib_.push_back(val);
	}
}

/*===========================================================================*/
TGnode::TGnode(int i, int t, unsigned ri, unsigned ru) :
	type(t),
	ri_(ri),
	ru_(ru),
	name(i),
	deadline(0.0),
	hard(false),		// is this right?
	h(0),
	w(0),
	attrib_(),
	series_children_(),
	series_parent_(-1)
{
	MAP(x, ArgPack::ap().task_attrib_av.size()) {
		double val = ArgPack::ap().task_attrib_av[x] +
		  RGen::gen().flat_range_d(-ArgPack::ap().task_attrib_mul[x],
		  ArgPack::ap().task_attrib_mul[x]);

		if (eps_not_equal_to<double>()(ArgPack::ap().task_attrib_round[x], 0.0)) {
			val = interval_round(val, ArgPack::ap().task_attrib_round[x]);
		}

		attrib_.push_back(val);
	}
}

/*###########################################################################*/
TG::TG() :
	maxw(0),
	maxh(0),
	low_bound_(-1),
	in_deg_(-1),
	out_deg_(-1),
	period_(-1.0),
	aperiod_(-1),
	number_(0),
	first_task_(0),
	series_arclist_(),
	num_start_nodes_(1)
{}

/*===========================================================================*/
void TG::init(int number, int in_deg_req, int out_deg_req, double period_req) {
	in_deg_ = in_deg_req;
	out_deg_ = out_deg_req;
	number_ = number;
	period_ = period_req;
}

/*===========================================================================*/
void TG::add_start_nodes () {
	bool multi_start = RGen::gen().flip(ArgPack::ap().prob_multi_start_nodes);
	if (multi_start) {
		num_start_nodes_ = ArgPack::ap().start_node_av + 
				RGen::gen().flat_range_l (-ArgPack::ap().start_node_mul, 
													ArgPack::ap().start_node_mul);
		if (num_start_nodes_ < 1) {
			num_start_nodes_ = 1;
		}
	}

	MAP (x, num_start_nodes_) {		
		add_vertex(TGnode(x));
	}
}
/*===========================================================================*/
void TG::init(int in_low_bound, int in_deg_req, int out_deg_req, int number,
int first_task) {
	low_bound_ = in_low_bound;
	in_deg_ = in_deg_req;
	out_deg_ = out_deg_req;
	number_ = number;
	first_task_ = first_task;

	add_start_nodes();
	if (ArgPack::ap().gen_series_parallel) {
		while (size_vertex() < low_bound_) {
			generate_series_parallel();
		}

		int local_xover = ArgPack::ap().series_local_xover;
		int global_xover = ArgPack::ap().series_global_xover;

		if (local_xover || global_xover) {
			series_parallel_generate_xover(local_xover, global_xover);
		}
	} else {
	  while (size_vertex() < low_bound_) {
		 augment();
	  }
	}
	compute_dims();

	deadlines();
}

/*===========================================================================*/
void TG::generate_series_parallel()
{
	int width = ArgPack::ap().series_wid_av;
	int wdev = ArgPack::ap().series_wid_mul;
	int length = ArgPack::ap().series_len_av;
	int ldev = ArgPack::ap().series_len_mul;
	if (wdev > 0)
		width = width + static_cast<int>(RGen::gen().flat_range_l(-wdev, wdev));

//	TGnode * source = FirstNode();
	long source = RGen::gen().flat_range_l(0, num_start_nodes_);
	while (int num_children = static_cast<int>((*this)[source].series_children_.size())) {
		int i;
		if (num_children > 1)
			i = static_cast<int>(RGen::gen().flat_range_l(0, num_children - 1));
		else
			i = 0;
	 	source = (*this)[source].series_children_[i];
	}

	long sync = 0; 
	if (ArgPack::ap().series_must_rejoin || 
		(ArgPack::ap().series_subgraph_fork_out<RGen::gen().flat_range_d(0,1))) 
	{
		if (vertex(source)->size_out()) {
			sync = edge(vertex(source)->out(0))->to();
			erase_edge(vertex(source)->out(0));
		} else {
			sync = add_vertex(TGnode(size_vertex()));
		}
	}

	MAPI (x, width) {
		long n = source;
		if (ldev > 0)
			length = length + static_cast<int>(RGen::gen().flat_range_l(-ldev, ldev));
		MAPI (y, length) {
			long new_node = size_vertex();
			add_vertex(TGnode(size_vertex()));
			(*this)[source].series_children_.push_back(new_node);
			(*this)[new_node].series_parent_ = source;
			add_edge(n, new_node, TGarc(static_cast<int>(size_edge())));

			n = new_node;
		}
		if (sync)
			add_edge(n, sync, TGarc(static_cast<int>(size_edge())));
	}
}

/*===========================================================================*/
void TG::series_parallel_generate_xover(int local_xover, int global_xover)
{
	RVector<long> tmp = series_parallel_get_all_parents();
	while (local_xover > 0) {
		random_shuffle(tmp.begin(), tmp.end(), RGen::gen());

		long p = -1;
		RVector<long>::iterator iter = tmp.begin(); 
		for ( ; iter != tmp.end(); iter++) {
			p = *iter;
			if ((*this)[p].series_children_.size() >= 4) {
				break;
			}
		}

		if (p != -1) {
			RVector<long> tmp2 = (*this)[p].series_children_;
			tmp2.push_back(p);
			random_shuffle(tmp2.begin(), tmp2.end(), RGen::gen());

			bool done = false;
			RVector<long>::iterator i = tmp2.begin(), j = tmp2.begin();
			++j;
			while(!done) {
				if (series_parallel_xarc_added (*i, *j)) {
					done = true;
				} else {
					j++;
					if (j == tmp2.end()) {
						j = i;
						i++;
					}
					if (i == tmp2.end()) {
						tmp.erase(iter);
						done = true;
					}
				}
			}
			--local_xover;
		} else {
			//global_xover += local_xover;
			local_xover = 0;
		}		
	}
	tmp.clear();
	MAP (x, size_vertex()) {
		tmp.push_back(x);
	}
	while (global_xover > 0) {
		random_shuffle(tmp.begin(), tmp.end(), RGen::gen());
		bool done = false;
		RVector<long>::iterator i = tmp.begin(), j = tmp.begin();
		++j;
		while(!done) {
			if (series_parallel_xarc_added (*i, *j)) {
				done = true;
			} else {
				j++;
				if (j == tmp.end()) {
					j = i;
					i++;
				}
				if (i == tmp.end()) {
					done = true;
					global_xover = 0;
				}
			}
		}
		--global_xover;
	}
}

RVector<long> TG::series_parallel_get_all_parents ()
{
	RVector<long> tmp;
	MAP (x, size_vertex()) {
		if ((*this)[x].series_children_.size()) {
			tmp.push_back(x);
		}
	}
	return tmp;
}

bool TG::series_parallel_xarc_added (long a, long b)
{
	if (nodes_linked (a, b)) {
		return false;		
	}

	add_edge (a, b, TGarc(static_cast<int>(size_edge())));

	if (cyclic()) {
		long t = vertex(a)->size_out();
		erase_edge(vertex(a)->out(t - 1));
		return false;
	}

	return true;
}

/*===========================================================================*/
void TG::augment() {
// SELECT TEMPLATE (either IN or OUT now)

	if (RGen::gen().flip()) {
// Pick FAN-OUT
		RVector<long>	nodes_with_most_fanout;
		int			max_fanout_found = 0;

		MAPI (x, size_vertex()) {
			long node_fanout_avail = out_deg_ - 
											static_cast<long>(vertex(x)->size_out());

			if( node_fanout_avail <= 0 ) continue;
			if( node_fanout_avail > max_fanout_found ) {
				max_fanout_found = static_cast<int>(node_fanout_avail);
				nodes_with_most_fanout.clear();
				nodes_with_most_fanout.insert(nodes_with_most_fanout.begin(),x);
//				nodes_with_most_fanout.push_back(x);
			}
			else if( node_fanout_avail == max_fanout_found ) {
				nodes_with_most_fanout.insert(nodes_with_most_fanout.begin(),x);
//				nodes_with_most_fanout.push_back(x);
			}
		}

		int nnodes = static_cast<int>(nodes_with_most_fanout.size());
		if( nnodes > 0 ) {
			// at least one has fanout room, and the list is those with
			// the most room, randomly select one
			int m = static_cast<int>(RGen::gen().flat_range_l(0, nnodes));

			// pick node
			long selected_node = nodes_with_most_fanout[m];
			int node_fanout_avail = out_deg_ - 
					static_cast<int>(vertex(selected_node)->size_out());

			// pick actual fanout value, including 0 as a bound
			int n = static_cast<int>(RGen::gen().flat_range_l(0, node_fanout_avail + 1));

			// add in the new nodes and arcs
			for (int x=0; x < n; x++) {
				add_vertex(TGnode(size_vertex()));
				add_edge(selected_node, static_cast<int>(size_vertex())-1, TGarc(static_cast<int>(size_edge())));
			}
		}
	}
	else {
// Pick FAN-IN
		// find out how many existing nodes are not over their
		// OUTDEG limit...
		int	nf = 0;
		MAP (x, size_vertex()) {
			if (vertex(x)->size_out() < out_deg_)
				nf++;
		}

// at least one
		RASSERT(nf);

		// determine # in deg
		int o = min(static_cast<int>(size_vertex()), min(in_deg_, nf));
		int n = static_cast<int>(RGen::gen().flat_range_l(1, o + 1));
		RGen::gen().flat_range_l(1, o + 1);

		// create new node
		add_vertex(TGnode(size_vertex()));

		// find places to attach, ensure no dupl arcs
		vertex_index node = static_cast<long>(size_vertex()) - 1;
		MAP (x, size_vertex()) {
			(*this)[x].h = (x == node ? 1 : 0);
		}

		RVector<long> indx;
		MAP(x, size_vertex()) {
			indx.push_back(x);
		}
		random_shuffle(indx.begin(), indx.end(), RGen::gen());

		long i = 0;
		int c = 0;

		while (c < n) {
			// randomly pick node, make sure not used before
			RASSERT( i < indx.size() );
			long trynode = indx[i++];

			if ((*this)[trynode].h) continue;
			if (static_cast<int>(vertex(trynode)->size_out()) >= out_deg_)
				continue;

			add_edge(trynode, node, TGarc(static_cast<int>(size_edge())));
			c++;
		}
	}
}

/*===========================================================================*/
void TG::print_to(ostream & os) const {
	if (period_ > 0.0) {
		os << "\tPERIOD " << period_ << "\n\n";
	} else {
	    if (ArgPack::ap().aperiodic_min_used) {
		os << "\tAPERIODIC " << aperiod_ << "\n\n";
	    } else {
		os << "\tAPERIODIC\n\n";
	    }
	}

	int indx = first_task_;
	MAP (x, size_vertex()) {
		os << "\tTASK t" << number_ << "_" << (*this)[x].name << "\tTYPE ";

		if ((*this)[x].type != -1) {
			os << (*this)[x].type << " ";
		} else if (ArgPack::ap().task_unique) {
			os << indx++ << " ";
		} else {
			os << RGen::gen().flat_range_l(0, ArgPack::ap().task_type_cnt) << " ";
		}

		MAP(y, ArgPack::ap().task_attrib_name.size()) {
			os << ArgPack::ap().task_attrib_name[y] << " " 
				<< (*this)[x].attrib_[y] << " ";
		}
		os << "\n";
	}

	os << "\n";

	MAP (x, size_edge()) {
		os << "\tARC "
			<< "a" << number_ << "_" << (*this)(x).name << " \tFROM "
			<< "t" << number_ << "_" << edge(x)->from() << "  TO  "
			<< "t" << number_ << "_" << edge(x)->to();

		int task_type;
		if( (*this)(x).type < 0 )
			task_type =
				static_cast<int>(RGen::gen().flat_range_l(0, ArgPack::ap().trans_type_cnt));
		else
			task_type = (*this)(x).type;

		os << " TYPE " << task_type << endl;
	}

	os << "\n";

// Deadlines
// SIMPLE FOR NOW: PUT DEADLINE ON ALL "TERMINAL PROCESSES"
	int dcount = 0;
	MAP (x, size_vertex()) {
		if((*this)[x].deadline > 0) {
			if ((*this)[x].hard) {
				os	<< "\tHARD_DEADLINE " <<
				  "d" << number_ << "_" << dcount++ <<
				  " ON " << "t" << number_ << 
					"_" <<(*this)[x].name << " AT " <<
				  (*this)[x].deadline << "\n";
			} else {
				os	<< "\tSOFT_DEADLINE " <<
				  "d" << number_ << "_" << dcount++ <<
				  " ON " << "t_" << (*this)[x].name << " AT " <<
				  (*this)[x].deadline * ArgPack::ap().soft_deadline_mul << "\n";
			}
		}
	}
	os << "}\n\n\n";
}

/*===========================================================================*/
void TG::push_period() {
// Find the max deadline.
	double max_d = 0.0;

	MAP (x, size_vertex()) {
		max_d = max(max_d, (*this)[x].deadline);
	}
	period_ = max(period_, max_d);
}

/*===========================================================================*/
void TG::compute_dims() {
    // find out how tall and wide the dag is ...
    // Get node depths, DUMB method [O(n^2) I think]
 
	MAP(x, size_vertex()) {
		(*this)[x].h = (*this)[x].w = -1;
	}
	int mh = 0;
	int change;
	do {
		change = 0;
		MAP(x, size_vertex()) {
			int pl = parents_h(x);
			if( pl+1 > ((*this)[x].h)) {
				(*this)[x].h = pl+1;
				mh = max(mh, (*this)[x].h);
				change = 1;
			}
		}
	} while(change);
	maxh = ++mh;   // mh is height

	// heights assigned, next widths
	RVector<int> wlist;
	MAP (x, mh) {
		wlist.push_back(0);
	}
	int mw = 0;
	MAP(x, size_vertex()) {
		(*this)[x].w = wlist[(*this)[x].h]++;
		mw = max(mw, (*this)[x].w);
	}
	maxw = ++mw;
}

/*===========================================================================*/
void TG::deadlines() {
// SIMPLE FOR NOW: PUT DEADLINE ON ALL "TERMINAL PROCESSES"
	MAP(x, size_vertex()) {
		if(!vertex(x)->size_out()) {
			const double dl = ((*this)[x].h + 1) * ArgPack::ap().vertex_time;

			const double jit = RGen::gen().flat_range_d(-1.0, 1.0) *
				ArgPack::ap().deadline_jitter * dl;

			(*this)[x].deadline = dl + jit;
		} else {
			(*this)[x].deadline = -1;
		}

		(*this)[x].hard = RGen::gen().flip(ArgPack::ap().prob_hard_deadline);
	}
}

/*===========================================================================*/
int TG::parents_h(long n) {
	int rv = -1;
	MAP (x, vertex(n)->size_in()) {
		edge_index e(vertex(n)->in(x));
		rv = max(rv, (*this)[edge(e)->from()].h);
	}
	return rv;
}

/*===========================================================================*/
void TG::print_to_vcg(ostream & os) const {
	int indx = first_task_;
	MAP(x, size_vertex()) {
		os << "\tnode: { title: \"" << "t" << number_ << "_" << (*this)[x].name
			<< "\" label: \"" << "t" << number_ << "_" << (*this)[x].name;
		if ((*this)[x].type != -1) {
			os << (*this)[x].type << "\" ";
		} else if (ArgPack::ap().task_unique) {
			os << " (" << indx++ << ")\" ";
		} else {
			os << " (" 
				<< RGen::gen().flat_range_l(0, ArgPack::ap().task_type_cnt)
				<< ")\" ";
		}

		if (vertex(x)->size_in()) {
			if (vertex(x)->size_out()) {
				os << " color: white ";
			} else {
				os << " color: lightred ";
			}
		} else {
			os << " color: lightgreen ";
		}
		os << "} \n";
	}

	os << "\n";

	MAP(x, size_edge()) {
		os << "\tedge: { thickness: 2 sourcename:\""
			<< "t" << number_ << "_" << (*this)[edge(x)->from()].name
			<< "\" targetname: \""
			<< "t" << number_ << "_" << (*this)[edge(x)->to()].name << "\" ";

		int task_type;
		if( (*this)(x).type < 0 )
			task_type =
				static_cast<int>(RGen::gen().flat_range_l(0, ArgPack::ap().trans_type_cnt));
		else
			task_type = (*this)(x).type;

		os << " label: \"(" << task_type << ")\" } \n" << endl;
	}
	os << "\n";
}

