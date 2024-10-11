// Copyright 2017 by Robert Dick, David Rhodes, and Keith Vallerio.
// All rights reserved.

#ifndef D_BASE_H_
#define D_BASE_H_

/*###########################################################################*/
#include "RString.h"
#include "RecVector.h"
#include "RVector.h"

//#include <cstddef>
//#include <iosfwd>
#include <string>

/*###########################################################################*/
class DBase {
public:
	DBase(int table_offset, int type_cnt);
	void print_to(std::ostream & os, const std::string & table_label) const;
	void trans_print_to(std::ostream &os, const std::string & table_label)const;

private:
		int type_cnt_;
		int table_offset_;

// PE, attribute
		rstd::RecVector2<double> table_;

// PE, type, attribute, different implementations
		rstd::RecVector3<rstd::RVector<double> > table_type_;
		const rstd::RVector<std::string> & table_name_;
		const rstd::RVector<std::string> & type_name_;

		int f_width_;
};

/*###########################################################################*/
#endif
