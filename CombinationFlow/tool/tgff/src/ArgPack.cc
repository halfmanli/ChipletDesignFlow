// Copyright 2017 by Robert Dick, David Rhodes, and Keith Vallerio.
// All rights reserved.

#include <cmath>
#include <functional>
#include <iomanip>
#include <map>

#include "RMath.h"
#include "RPair.h"
#include "RVector.h"
#include "RecVector.h"
#include "ArgPack.h"
#include "RStd.h"
#include "PGraph.h"
#include "TGraph.h"
#include "RGen.h"
#include "DBase.h"

using namespace std;
using namespace rstd;

/*###########################################################################*/
// Private
ArgPack * ArgPack::def_ap_ = 0;

/*===========================================================================*/
// Public
/*===========================================================================*/
ArgPack::ArgPack(int argc, char * const argv[]) :
	tg_label("TASK_GRAPH"),
	tg_cnt(3),
	vertex_time(100),
	deadline_jitter(0.0),
	vertex_cnt_av(20),
	vertex_cnt_mul(5),
	vertex_in_deg(5),
	vertex_out_deg(5),
	task_unique(false),
	task_type_cnt(static_cast<int>((tg_cnt * vertex_cnt_av) / 3)),
	trans_unique(false),
	trans_type_cnt(static_cast<int>((tg_cnt * vertex_cnt_av *
		(vertex_in_deg + vertex_out_deg)) / 12)),
	task_attrib_name(),
	task_attrib_av(),
	task_attrib_mul(),
	task_attrib_round(),
	prob_multi_start_nodes(0.0),
	start_node_av(2),
	start_node_mul(1),
	p_laxity(1.0),
	p_greater_deadline(false),
	prob_hard_deadline(1.0),
	soft_deadline_mul(1.0),
	table_label("PE"),
	table_cnt(5),
	type_table_ratio(0.5),
	period_mul(3),
	entries_per_type_av(1.0),
	entries_per_type_mul(0.0),
	table_name(1),
	table_av(1),
	table_mul(1),
	table_jitter(1),
	table_round(1),
	type_name(1),
	type_av(1),
	type_mul(1),
	type_jitter(1),
	type_round(1),
	gen_series_parallel(false),
   series_subgraph_fork_out(0),
	series_must_rejoin(false),
	series_len_av(5),
	series_len_mul(1),
	series_wid_av(2),
	series_wid_mul(1),
	series_local_xover (0),
	series_global_xover (0),
	seed(0),
	misc_type_cnt(0),
	prob_periodic(1.0),
	aperiodic_min_used (false),
	aperiodic_min_av(10),
	aperiodic_min_mul(5),
	data_file_name("tgff.tgff"),
	eps_file_name("tgff.eps"),
	vcg_file_name("tgff.vcg"),
	opt_file_name("tgff.tgffopt"),
	data_out_(),
	eps_out_(),
	vcg_out_(),
	opt_in_()
{
	RASSERT(! def_ap_);
	def_ap_ = this;

	period_mul[0] = 0.5;
	period_mul[1] = 1.0;
	period_mul[2] = 2.0;

	table_name[0] = "price";
	table_av[0] = 10.0;
	table_mul[0] = 5.0;
	table_jitter[0] = 0.5;
	table_round[0] = 0.0;

	type_name[0] = "exec_time";
	type_av[0] = 100.0;
	type_mul[0] = 40.0;
	type_jitter[0] = 0.5;
	type_round[0] = 0.0;

	if (argc != 2) {
		cout << help_;
		exit(EXIT_FAILURE);
	}

	data_file_name = string(argv[1]) + ".tgff";
	eps_file_name = string(argv[1]) + ".eps";
	vcg_file_name = string(argv[1]) + ".vcg";
	opt_file_name = string(argv[1]) + ".tgffopt";

	opt_in_.open(opt_file_name.c_str());
	if (! opt_in_) {
		cout << "Unable to open input file.\n";
		exit(EXIT_FAILURE);
	}

	eps_out_.open(eps_file_name.c_str());
	if (! eps_out_) {
		cout << "Unable to open EPS output file.\n";
		exit(EXIT_FAILURE);
	}

	vcg_out_.open(vcg_file_name.c_str());
	if (! vcg_out_) {
		cout << "Unable to open VCG output file.\n";
		exit(EXIT_FAILURE);
	} else {
		vcg_out_ << "graph: { label: \"" << vcg_file_name << "\"\n";
		vcg_out_ << "display_edge_labels: yes\n";
	}

	double h_period = parse();

	ofstream real_out(data_file_name.c_str());
	if (! real_out) {
		cout << "Unable to open TGFF output files.\n";
		exit(EXIT_FAILURE);
	}

	if (h_period > 0.0) {
		real_out << "@HYPERPERIOD " << h_period << "\n\n";
	}

	data_out_ << endl;

// Copy the output buffer to the real output file.
	real_out << data_out_.str();

	vcg_out_ << "\n } \n";
}

/*===========================================================================*/
ArgPack::~ArgPack() {
	RASSERT(def_ap_);
	def_ap_ = 0;
}

/*===========================================================================*/
// Private
/*===========================================================================*/
double
ArgPack::parse() {
	RVector<double> h_period;

	int line = 0;
	map<string, long> tg_offset;
	map<string, int> table_offset;

	tg_offset.insert(make_pair(tg_label, 0));
	table_offset.insert(make_pair(table_label, 0));

	while (1) {
		string s;
		bool wrap = true;

// Get a single command
		do {
			string line_buf;

			if (! getline(opt_in_, line_buf)) {
				if (h_period.size()) {
					return lcm(h_period);
				} else {
					return -1.0;
				}
			}

			line++;

			RVector<string> tokenized = tokenize(line_buf);
			
			if (! tokenized.empty() &&
				tokenized[0][0] != '#')
			{
				s += line_buf;

				if (! s.empty() && s[s.size() - 1] == '\\')
					s.erase(s.size() - 1);
				else
					wrap = false;
			}
		} while (wrap);

		string first = first_token(s);
		if (first.empty())
			parse_error(line);

		const string command = first;
		pop_token(s);

		RVector<string> vec = tokenize(s);
		RVector<string> top_rvec(0);
		if (! vec.empty())
			top_rvec = tokenize(s, ",");

		MVector<2, string> rvec(top_rvec.size());

		MAP(x, rvec.size()) {
			rvec[x] = tokenize(top_rvec[x]);
		}

/*-------------------------------------*/
		if (command == "tg_label") {
			if (vec.size() != 1) parse_error(line);
			tg_label = vec[0];
			tg_offset.insert(make_pair(tg_label, 0));
		} else if (command == "tg_offset") {
			if (vec.size() != 2) parse_error(line);
			const string & tg_name = vec[0];
			const long offset = Conv(vec[1]);
			tg_offset[tg_name] = offset;
		} else if (command == "tg_cnt") {
			if (vec.size() != 1 ||
			  (tg_cnt = Conv(vec[0])) < 0)
				parse_error(line);

		} else if (command == "task_trans_time") {
			if (vec.size() != 1 ||
			  (vertex_time = Conv(vec[0])) < 0)
				parse_error(line);

		} else if (command == "deadline_jitter") {
			if (vec.size() != 1) parse_error(line);
			deadline_jitter = Conv(vec[0]);
			if (deadline_jitter < 0.0 || deadline_jitter >= 1.0) {
				parse_error(line);
			}

		} else if (command == "pack_schedule") {
			if ( !(vec.size() == 10 || vec.size() == 11) )
				parse_error(line);

			int		num_task_graphs	 = Conv(vec[0]);
			int		avg_tasks_per_pe = Conv(vec[1]);
			if( num_task_graphs < 1 || avg_tasks_per_pe < 1)
				parse_error(line);

			double	avg_task_time	= Conv(vec[2]);
			double	mul_task_time	= Conv(vec[3]);
			double	task_slack		= Conv(vec[4]);
			double	task_round		= Conv(vec[5]);
			int		num_pe_types	= Conv(vec[6]);
			int		num_pe_soln		= Conv(vec[7]);
			if( avg_task_time <= 0.0 ||
				avg_task_time+abs(mul_task_time) <= 0.0 ||
				task_slack < 0.0 ||
				num_pe_types < 1 ||
				num_pe_soln < 1 )
				parse_error(line);

			int		num_com_types	= Conv(vec[8]);
			int		num_com_soln	= Conv(vec[9]);
			if( num_com_types < 1 ||
				num_com_soln < 1 )
				parse_error(line);
			double	arc_fill_factor	= 1.0;
			if( vec.size() == 11 )
				arc_fill_factor	= Conv(vec[10]);

			// here goes...
			PGraph pg(	eps_out_,
						num_task_graphs,
						avg_tasks_per_pe,
						avg_task_time,	/* PE STUFF */
						mul_task_time,
						task_slack,
						task_round,
						num_pe_types,
						num_pe_soln,
						num_com_types,	/* COM STUFF */
						num_com_soln,
						arc_fill_factor);

			pg.print_to(data_out_);
			return pg.h_period();

		} else if (command == "task_cnt") {
			if (vec.size() != 2) parse_error(line);
			vertex_cnt_av = Conv(vec[0]);
			vertex_cnt_mul = Conv(vec[1]);

			if (vertex_cnt_av - abs(vertex_cnt_mul) < 0) {
				parse_error(line);
			}

		} else if (command == "task_degree") {
			if (vec.size() != 2) parse_error(line);
			vertex_in_deg = Conv(vec[0]);
			vertex_out_deg = Conv(vec[1]);

			/* when using "pack_schedule" -1 indicates "no limit"
			if (vertex_in_deg_ < 1 || vertex_out_deg_ < 1)
				parse_error(line);
			*/

		} else if (command == "task_unique") {
			if (vec.size() > 1) parse_error(line);
			if (vec.size())
				task_unique = Conv(vec[0]);
			else
				task_unique = true;

		} else if (command == "prob_multi_start_nodes") {
			if (vec.size() != 1) parse_error(line);
			prob_multi_start_nodes = Conv(vec[0]);

		} else if (command == "start_node") {
			if (vec.size() != 2) parse_error(line);
			start_node_av = Conv(vec[0]);
			start_node_mul = Conv(vec[1]);
			if (start_node_av < 1) parse_error(line);

		} else if (command == "task_type_cnt") {
			if (vec.size() != 1 ||
			  (task_type_cnt = Conv(vec[0])) < 0)
				parse_error(line);

		} else if (command == "trans_unique") {
			if (vec.size() > 1) parse_error(line);
			if (vec.size() == 1)
				trans_unique = Conv(vec[0]);
			else
				trans_unique = true;

		} else if (command == "trans_type_cnt") {
			if (vec.size() != 1 ||
			  (trans_type_cnt = Conv(vec[0])) < 0)
				parse_error(line);

		} else if (command == "task_attrib") {
			task_attrib_name.clear();
			task_attrib_av.clear();
			task_attrib_mul.clear();
			task_attrib_round.clear();

			MAP(x, rvec.size()) {
				if (rvec[x].size() < 3 || rvec[x].size() > 4)
					parse_error(line);

				task_attrib_name.push_back(rvec[x][0]);
				task_attrib_av.push_back(Conv(rvec[x][1]));
				task_attrib_mul.push_back(Conv(rvec[x][2]));
	
				if (rvec[x].size() >= 4) {
					task_attrib_round.push_back(Conv(rvec[x][3]));
				} else {
					task_attrib_round.push_back(0.0);
				}
			}

		} else if (command == "period_laxity") {
			if (vec.size() != 1 ||
				(p_laxity = Conv(vec[0])) <= 0.0)
			{
				parse_error(line);
			}

		} else if (command == "period_g_deadline") {
			if (vec.size() > 1) parse_error(line);
			if (vec.size() == 1) {
				p_greater_deadline = Conv(vec[0]);
			} else {
				p_greater_deadline = true;
			}

		} else if (command == "prob_hard_deadline") {
			if (vec.size() > 1) parse_error(line);
			if (vec.size() == 1)
				prob_hard_deadline = Conv(vec[0]);

			if (prob_hard_deadline < 0.0 ||
				prob_hard_deadline > 1.0)
			{
				parse_error(line);
			}

		} else if (command == "soft_deadline_mul") {
			if (vec.size() > 1) parse_error(line);
			if (vec.size() == 1) {
				soft_deadline_mul = Conv(vec[0]);
			}

			if (soft_deadline_mul < 0.0) {
				parse_error(line);
			}
			
		} else if (command == "period_mul") {
			period_mul.clear();
			MAP(x, rvec.size()) {
				if (rvec[x].size() != 1) {
					parse_error(line);
				}

				period_mul.push_back(Conv(rvec[x][0]));

				if (period_mul.back() < 0.0) {
					parse_error(line);
				}
			}

		} else if (command == "entries_per_type") {
			if (vec.size() != 2) parse_error(line);
			entries_per_type_av = Conv(vec[0]);
			entries_per_type_mul = Conv(vec[1]);

			if (rint(entries_per_type_av -
				abs(entries_per_type_mul)) < 0.0)
			{
				parse_error(line);
			}


		} else if (command == "table_label") {
			if (vec.size() != 1) parse_error(line);
			table_label = vec[0];
			table_offset.insert(make_pair(table_label, 0));

		} else if (command == "table_cnt") {
			if (vec.size() != 1 ||
			  (table_cnt = Conv(vec[0])) < 0)
				parse_error(line);

		} else if (command == "type_table_ratio") {
			if (vec.size() != 1 ||
			  (type_table_ratio = Conv(vec[0])) < 0)
				parse_error(line);

		} else if (command == "table_attrib") {
			table_name.clear();
			table_av.clear();
			table_mul.clear();
			table_jitter.clear();
			table_round.clear();

			MAP(x, rvec.size()) {
				if (rvec[x].size() < 3 || rvec[x].size() > 5)
					parse_error(line);

				table_name.push_back(rvec[x][0]);
				table_av.push_back(Conv(rvec[x][1]));
				table_mul.push_back(Conv(rvec[x][2]));
	
				if (rvec[x].size() >= 4) {
					table_jitter.push_back(Conv(rvec[x][3]));
				} else {
					table_jitter.push_back(0.5);
				}

				if (rvec[x].size() == 5) {
					table_round.push_back(Conv(rvec[x][4]));
				} else {
					table_round.push_back(0.0);
				}
			}

		} else if (command == "type_attrib") {
			type_name.clear();
			type_av.clear();
			type_mul.clear();
			type_jitter.clear();
			type_round.clear();

			MAP(x, rvec.size()) {
				if (rvec[x].size() < 3 || rvec[x].size() > 5)
					parse_error(line);

				type_name.push_back(rvec[x][0]);
				type_av.push_back(Conv(rvec[x][1]));
				type_mul.push_back(Conv(rvec[x][2]));

				if (rvec[x].size() >= 4) {
					type_jitter.push_back(Conv(rvec[x][3]));
				} else {
					type_jitter.push_back(0.5);
				}

				if (rvec[x].size() == 5) {
					type_round.push_back(Conv(rvec[x][4]));
				} else {
					type_round.push_back(0.0);
				}
			}

		} else if (command == "table_offset") {
			if (vec.size() != 2) parse_error(line);
			const string & pe_name = vec[0];
			const int offset = Conv(vec[1]);
			table_offset[pe_name] = offset;
		} else if (command == "seed") {
			if (vec.size() != 1) parse_error(line);
			seed = Conv(vec[0]);


		} else if (command == "gen_series_parallel") {
			if (vec.size() > 1) parse_error(line);
			if (vec.size()) {
				gen_series_parallel = Conv(vec[0]);
			} else {
				gen_series_parallel = true;
			}
		} else if (command == "series_must_rejoin") {
			if (vec.size() > 1) parse_error(line);
			if (vec.size())
				series_must_rejoin = Conv(vec[0]);
			else
				series_must_rejoin = true;
		} else if (command == "series_subgraph_fork_out") {
			if (vec.size() != 1) parse_error(line);
			series_subgraph_fork_out = Conv(vec[0]);

		} else if (command == "series_len") {
			if (vec.size() != 2) parse_error(line);
			series_len_av = Conv(vec[0]);
			series_len_mul = Conv(vec[1]);
		} else if (command == "series_wid") {
			if (vec.size() != 2) parse_error(line);
			series_wid_av = Conv(vec[0]);
			series_wid_mul = Conv(vec[1]);

		} else if (command == "series_local_xover") {
			if (vec.size() != 1) parse_error(line);
			series_local_xover = Conv(vec[0]);
		} else if (command == "series_global_xover") {
			if (vec.size() != 1) parse_error(line);
			series_global_xover = Conv(vec[0]);


		} else if (command == "misc_type_cnt") {
			if (vec.size() != 1) parse_error(line);
			misc_type_cnt = Conv(vec[0]);
		} else if (command == "prob_periodic") {
		  if (vec.size() != 1) parse_error(line);
		  prob_periodic = Conv(vec[0]);
		  if (prob_periodic < 0.0 || prob_periodic > 1.0) {
			 parse_error(line);
		  }
		} else if (command == "aperiodic_min") {
			if (vec.size() != 2) parse_error(line);
			aperiodic_min_used = true;
			aperiodic_min_av = Conv(vec[0]);
			aperiodic_min_mul = Conv(vec[1]);


		} else if (command == "note_write") {
			MAP(x, vec.size())
				data_out_ << vec[x] << " ";

			data_out_ << endl;

		} else if (command == "eps_write") {
			RGen::gen().set_seed(seed);
			TGraph tg(tg_offset[tg_label]);
			tg.print_to_ps(eps_out_, eps_file_name);

		} else if (command == "vcg_write") {
			RGen::gen().set_seed(seed);
			TGraph tg(tg_offset[tg_label]);
			tg.print_to_vcg(vcg_out_);

		} else if (command == "vcg_hide_edge_labels") {
			vcg_out_ << "display_edge_labels: no\n";

		} else if (command == "tg_write") {
			RGen::gen().set_seed(seed);
			TGraph tg(tg_offset[tg_label]);
			tg_offset[tg_label] += tg_cnt;
			tg.print_to(data_out_);
			h_period.push_back(tg.h_period());

		} else if (command == "pe_write") {
			RGen::gen().set_seed(seed);
			DBase db(table_offset[table_label], task_type_cnt);
			table_offset[table_label] += table_cnt;
			db.print_to(data_out_, table_label);

		} else if (command == "trans_write") {
			RGen::gen().set_seed(seed);

// ugly hack
			double old_av = entries_per_type_av;
			entries_per_type_av = 1.0;

			double old_mul = entries_per_type_mul;
			entries_per_type_mul = 0.0;

			DBase db(table_offset[table_label], trans_type_cnt);
			table_offset[table_label] += table_cnt;
			db.trans_print_to(data_out_, table_label);

			entries_per_type_av = old_av;
			entries_per_type_mul = old_mul;
		} else if (command == "misc_write") {
			RGen::gen().set_seed(seed);
			DBase db(table_offset[table_label], misc_type_cnt);
			table_offset[table_label] += table_cnt;
			db.print_to(data_out_, table_label);

		} else if (command == "opt_write") {
//			ifstream is(opt_file_name.begin());
			ifstream is;
			is.open(opt_file_name.c_str());

			if (! is) parse_error(line);
	
			stringbuf tmp;
			do {
				if (! is.get(tmp)) {
					parse_error(line);
				}

				data_out_ << "# " << tmp.str();
			} while (is);

		} else {
			parse_error(line);
		}
	}
}

/*===========================================================================*/
void ArgPack::parse_error(int line) {
	cout << "Parse error on line " << line << ".\n";
	exit(EXIT_FAILURE);
}

/*===========================================================================*/
const char * ArgPack::help_ =
"tgff [filename]\n"
"    reads commands from the [filename].tgffopt file\n"
"    writes PostScript to the [filename].eps file\n"
"    writes VCG readable graph to the [filename].vcg file\n"
"    writes data to the [filename].tgff file\n"
"\n"
"A \\ can be used to enter multi-line commands.\n"
"A # at the start of a line comments out the line.\n"
"Multipliers indicate values which are used to scale a random number [-1,1).\n"
"\n"
"TGFF understands the following data types:\n"
"------------------------------------------\n"
"  <string>: a string of characters\n"
"  <int>: an integer number\n"
"  <flt>: a floating point number\n"
"  <bool>: a boolean value which can be ``true'' or ``false''\n"
"  <list(T)>: a comma separated series of zero or more values of type T\n"
"    parentheses are a meta-character, and are not entered in the input file\n"
"\n"
"TGFF understands the following variable setting commands:\n"
"---------------------------------------------------------\n"
"  seed <int>: seeds the pseudo-random number generator\n"
"  tg_label <string>: label used for task graphs\n"
"  tg_offset <string> <int>: sets index for named task graph\n"
"  tg_cnt <int>: number of task graphs\n"
"  task_trans_time <flt>: average time per task including communication\n"
"    (used in setting deadlines)\n"
"  deadline_jitter <flt>: proportional jitter for deadline\n"
"  task_cnt <int> <int>: number of tasks per graph (average, multiplier)\n"
"  task_degree <int> <int>: maximum number of transmits (in, out) per task\n"
"  task_unique <bool>: tasks types are forced to be unique (true by default) \n"
"  task_type_cnt <int>: number of task types \n"
"  trans_type_cnt <int>: number of transmit types\n"
"  task_attrib <list(<string> <flt> <flt> <flt>)>:\n"
"    name, average, multiplier, round to (default 0.0, 0.0 means no rounding)\n"
"  period_laxity <flt>: laxity of periods, relative to deadlines (default 1)\n"
"  period_g_deadline <bool>: periods forced > deadlines (default true)\n"
"  prob_hard_deadline <flt>: probability that a deadline will be hard (vs. soft)\n"
"  soft_deadline_mul <flt>: multiplier applied to soft deadlines (default 1)\n"
"  period_mul <list(<int>)>: multipliers for periods in multirate systems\n"
"    Multipliers randomly selected from this list.\n"
"  prob_periodic <flt>: probability that a graph is periodic (default 1.0)\n"
"  prob_multi_start_nodes <flt>: probability that a graph has more than one\n"
"    start nodes (default 0.0)\n"
"  start_node <int> <int>: number of start nodes for graphs which have\n"
"    multiple start nodes (average, multiplier)\n"
"\n"
"TGFF series-parallel graph constuction variables:\n"
"---------------------------------------------------------\n"
"  gen_series_parallel <bool>: generate graphs with a series-parallel structure\n"
"  series_must_rejoin <bool>: forces subgraphs formed in series chains to\n"
"    rejoin into the main graph (overrides series_subgraph_fork_out flag)\n"
"  series_subgraph_fork_out <flt>: probability subgraphs will not rejoin\n"
"  series_len <int> <int>: length of series chains (average, multiplier)\n"
"  series_wid <int> <int>: width of series chains (average, multiplier)\n"
"  series_local_xover <int>: number of extra local arcs added\n"
"  series_global_xover <int>: number of extra global arcs added\n"
"\n"
"TGFF table construction variables\n"
"---------------------------------------------------------\n"
"  entries_per_type <flt> <flt>:\n"
"  number of attribute entries per task type for tables (average, multiplier)\n"
"  table_label <string>: label used for tables\n"
"  table_offset <string> <int>: sets index for named table\n"
"  table_cnt <int>: number of tables\n"
"  type_table_ratio <flt>: ratio of statistical contributions\n"
"                          of type to table attributes (default 0.5)\n"
"\n"
"  table_attrib <list(<string> <flt> <flt> <flt> <flt>)>:\n"
"    name, average, multiplier, jitter (default 0.5),\n"
"    round to (default 0.0, 0.0 means no rounding)\n"
"\n"
"  type_attrib <list(<string> <flt> <flt> <flt> <flt>)>:\n"
"    name, average, multiplier, jitter (default 0.5),\n"
"    round to (default 0.0, 0.0 means no rounding)\n"
"\n"
"TGFF understands the following general commands:\n"
"------------------------------------------------\n"
"  vcg_write: creates an input file for the graph visulization tool VCG\n"
"    [to .vcg  file]\n"
"  vcg_hide_edge_labels: suppresses display of edge labels  [for .vcg file]\n"
"  eps_write: make a PostScript plot of the task graphs   [to .eps  file]\n"
"  tg_write: write the task graphs                        [to .tgff file]\n"
"  pe_write: write PE information                         [to .tgff file]\n"
"  trans_write: write transmission event information      [to .tgff file]\n"
"  misc_write: write independant processor information    [to .tgff file]\n"
"  misc_type_cnt: number of types for misc_write\n"
"  note_write <list(<string>)>: write the string(s)       [to .tgff file]\n"
"\n"
"  pack_schedule <int> <int> <flt> <flt> <flt> <flt> ...  [to .tgff file]\n"
"            ... <int> <int>  <int> <int> [<flt>]         [&  .eps file]\n"
"           [... on one line]\n"
"  ** this is a `self-contained' command, it generates PEs, COMs, TG, etc.\n"
"     other writes except note_write should not be used. Only non-periodic\n"
"     instances are generated, 'task_degree' and 'seed' are only cmds that \n"
"     affect its operation\n"
"  ** args in-order are:\n"
"     num_task_graphs avg_task_graphs_per_pe\n"
"     avg_task_time mul_task_time task_slack task_round num_pe_types num_pe_soln\n"
"     num_com_types num_com_soln and optionally arc_fill_factor\n"
"\n";
