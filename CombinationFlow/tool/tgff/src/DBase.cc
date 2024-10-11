// Copyright 2017 by Robert Dick, David Rhodes, and Keith Vallerio.
// All rights reserved.

#include "DBase.h"
#include "ArgPack.h"
#include "RGen.h"
#include "Epsilon.h"
#include "RMath.h"

#include <cmath>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <algorithm>

using namespace std;
using namespace rstd;

/*###########################################################################*/
DBase::DBase(int table_offset, int type_cnt) :
	type_cnt_(type_cnt),
	table_offset_(table_offset),
	table_(ArgPack::ap().table_cnt, ArgPack::ap().table_av.size()),

	table_type_(ArgPack::ap().table_cnt, type_cnt_, ArgPack::ap().type_av.size()),

	table_name_(ArgPack::ap().table_name),
	type_name_(ArgPack::ap().type_name),
	f_width_(0)
{
	RVector<double> table_stream(ArgPack::ap().table_cnt);
	MAP(x, table_stream.size()) {
		table_stream[x] = RGen::gen().flat_range_d(-1.0, 1.0);
	}

	RVector<double> type_stream(type_cnt_);
	MAP(x, type_stream.size()) {
		type_stream[x] = RGen::gen().flat_range_d(-1.0, 1.0);
	}

/*-------------------------------------*/
	MAP(x, table_name_.size()) {
		f_width_ = max(f_width_, static_cast<int>(table_name_[x].size()) + 2);
	}

	MAP(x, type_name_.size()) {
		f_width_ = max(f_width_, static_cast<int>(type_name_[x].size()) + 2);
	}

	MAP2(x, table_.size()[0], y, table_.size()[1]) {
		double jit = RGen::gen().flat_range_d(-1.0, 1.0);

		double deviation =
			flat_merge(ArgPack::ap().table_jitter[y], jit, table_stream[x]);

		deviation *= ArgPack::ap().table_mul[y];

		table_(x, y) = ArgPack::ap().table_av[y] + deviation;

		if (eps_not_equal_to<double>()(ArgPack::ap().table_round[y], 0.0)) {
			table_(x, y) =
			  interval_round(table_(x, y), ArgPack::ap().table_round[y]);
		}
	}

// x: table, y: type, z: attribute
	MAP2(x, table_type_.size()[0], y, table_type_.size()[1]) {
		long imp_cnt = static_cast<long>(rint(
			ArgPack::ap().entries_per_type_av +
			ArgPack::ap().entries_per_type_mul *
			RGen::gen().flat_range_d(-1.0, 1.0)));

		MAP2(imp, imp_cnt, z, table_type_.size()[2]) {
			double deviation = flat_merge(ArgPack::ap().type_table_ratio,
				type_stream[y], table_stream[x]);

			double jit = RGen::gen().flat_range_d(-1.0, 1.0);

			deviation = flat_merge(ArgPack::ap().type_jitter[z], jit, deviation);

			deviation *= ArgPack::ap().type_mul[z];

	 		table_type_(x, y, z).push_back(ArgPack::ap().type_av[z] + deviation);

			if (eps_not_equal_to<double>()(ArgPack::ap().type_round[z], 0.0)) {
				table_type_(x, y, z).back() =
				  interval_round(table_type_(x, y, z).back(),
				  ArgPack::ap().type_round[z]);
			}
		}
	}
}

/*===========================================================================*/
void DBase::print_to(ostream & os, const string & table_label) const {
// Print out the tables.

// FIXME: Library bug breaks string setw().  Use c_str() hack.

	os.setf(ios::left);

	MAP(x, table_.size()[0]) {
		os << "@" << table_label << " " <<
		  (x + table_offset_) << " {\n";

		if (table_name_.size()) {
			os << "# ";

			MAP(y, table_name_.size()) {
				if (y != table_name_.size() - 1) {
					os << setw(f_width_);
				}

				os << table_name_[y].c_str();
			}

			os << "\n  ";

			MAP(y, table_.size()[1]) {
				if (y != table_.size()[1] - 1) {
					os << setw(f_width_);
				}

				os << table_(x, y);
			}

			os << "\n";
		}

		if (type_cnt_ && table_name_.size()) {
			os << "\n#----------------------------------------" <<
			  "--------------------------------------\n";
		}

		if (type_cnt_) {
			os << "# " << "type version ";
			MAP(y, type_name_.size()) {
				if (y != type_name_.size() - 1)
					os << setw(f_width_);

				os << type_name_[y].c_str();
			}

			os << "\n";

// All types, all implementations
			MAP2(y, table_type_.size()[1],
				implemen, table_type_(x, y, 0).size())
			{
				os << setw(2) << "  " << setw(static_cast<int>(strlen("type "))) << y <<
				  setw(static_cast<int>(strlen("version "))) << implemen;

// implementations, and attributes.
				MAP(z, table_type_.size()[2]) {
					if (z != table_type_.size()[2] - 1) {
						os << setw(f_width_);
					}

					os << table_type_(x, y, z)[implemen];
				}
				os << "\n";
			}
		}

		os << "}\n\n\n";
	}

	os << "\n";

	os.clear();
}


/*===========================================================================*/
void DBase::trans_print_to(ostream & os, const string & table_label) const {
// Print out the tables.
	MAP(x, table_.size()[0]) {
		os << "@" << table_label << " " <<
		  (x + table_offset_) << " {\n";

		if (table_name_.size()) {
			os << "# ";

			MAP(y, table_name_.size()) {
				os << " " << setw(f_width_) << table_name_[y].c_str();
			}

			os << "\n  ";

			MAP(y, table_.size()[1]) {
				os << " " << setw(f_width_) << table_(x, y);
			}

			os << "\n";
		}

		if (type_cnt_ && table_name_.size()) {
			os << "\n#----------------------------------------" <<
			  "--------------------------------------\n";
		}

		if (type_cnt_) {
			os << "# " << "type ";
			MAP(y, type_name_.size()) {
				os << " " << setw(f_width_) << type_name_[y].c_str();
			}

			os << "\n";

// All types
			MAP(y, table_type_.size()[1]) {
				os << "  " << setw(static_cast<int>(strlen("type "))) << y;
// attributes.
				MAP(z, table_type_.size()[2]) {
					RASSERT(table_type_(x, y, z).size() == 1);
					os << " " << setw(f_width_) << table_type_(x, y, z).back();
				}
				os << "\n";
			}
		}

		os << "}\n\n\n";
	}

	os << "\n";
}



