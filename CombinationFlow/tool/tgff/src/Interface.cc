// Copyright 2017 by Robert Dick.
// All rights reserved.

#include "Interface.h"
#include <set>

namespace rstd {
using namespace std;

/*===========================================================================*/
ostream &
operator<<(ostream & os, comp_type c) {
	switch (c) {
	case LESS:
		os << "less";
		break;
	case GREATER:
		os << "greater";
		break;
	case EQ:
		os << "equal";
		break;
	default:
		Rabort();
	}

	return os;
}

void Interface_test() {
	const set<int> a;
	const set<int> b;
	int y = a < b;
	y++;
}

}
