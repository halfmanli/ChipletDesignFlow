// Copyright 2017 by Robert Dick, David Rhodes, and Keith Vallerio.
// All rights reserved.

#ifndef ARG_PACK_H_
#define ARG_PACK_H_

/*###########################################################################*/
#include "RVector.h"
#include "RStd.h"

#include <string>
#include <sstream>
#include <fstream>

/*===========================================================================*/
class ArgPack {
public:
// Holds all arguments to tgff. 

	static const ArgPack & ap() { RASSERT(def_ap_); return *def_ap_; }
	static ArgPack & write_ap() { RASSERT(def_ap_); return *def_ap_; }

	ArgPack(int argc, char * const argv[]);
	~ArgPack();

		std::string tg_label;
		int tg_cnt;
		double vertex_time;

		double deadline_jitter;

		double vertex_cnt_av;
		double vertex_cnt_mul;

		int vertex_in_deg;
		int vertex_out_deg;

		bool task_unique;
		int task_type_cnt;

		bool trans_unique;
		int trans_type_cnt;

		rstd::RVector<std::string> task_attrib_name;
		rstd::RVector<double> task_attrib_av;
		rstd::RVector<double> task_attrib_mul;
		rstd::RVector<double> task_attrib_round;

		double prob_multi_start_nodes;
		int start_node_av;
		int start_node_mul;

		double p_laxity;
		bool p_greater_deadline;
		double prob_hard_deadline;
		double soft_deadline_mul;

		std::string table_label;
		int table_cnt;

		double type_table_ratio;

		rstd::RVector<double> period_mul;

		double entries_per_type_av;
		double entries_per_type_mul;

		rstd::RVector<std::string> table_name;
		rstd::RVector<double> table_av;
		rstd::RVector<double> table_mul;
		rstd::RVector<double> table_jitter;
		rstd::RVector<double> table_round;

		rstd::RVector<std::string> type_name;
		rstd::RVector<double> type_av;
		rstd::RVector<double> type_mul;
		rstd::RVector<double> type_jitter;
		rstd::RVector<double> type_round;

		bool gen_series_parallel;
		double series_subgraph_fork_out;
		bool series_must_rejoin;
		int series_len_av;
		int series_len_mul;
		int series_wid_av;
		int series_wid_mul;
		int series_local_xover;
		int series_global_xover;

		int seed;
		int misc_type_cnt;
		double prob_periodic;
		bool aperiodic_min_used;
		long aperiodic_min_av;
		long aperiodic_min_mul;

		std::string data_file_name;
		std::string eps_file_name;
		std::string vcg_file_name;
		std::string opt_file_name;

private:
// Returns hyperperiod.
	double parse();
	void parse_error(int line);

		static ArgPack * def_ap_;
		static const char * help_;

		std::ostringstream data_out_;
		std::ofstream eps_out_;
		std::ofstream vcg_out_;
		std::ifstream opt_in_;
};

/*###########################################################################*/
#endif

