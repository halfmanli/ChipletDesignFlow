// Copyright 2017 by Robert Dick.
// All rights reserved.

#include "RStd.h"

#include <iostream>
#include <cstdlib>

namespace rstd {
using namespace std;

/*###########################################################################*/
ostream cerrhex(cerr.rdbuf());
ostream couthex(cout.rdbuf());

namespace {
// Cause hex streams to be initialized.
	int cerr_hex_init() { cerrhex << hex; return 0; }
	int cout_hex_init() { couthex << hex; return 0; }

	int L_cout_hex_init = cout_hex_init();
	int L_cerr_hex_init = cerr_hex_init();
}

/*===========================================================================*/
void rpos(const char * file, int line) {
	cerr << "File: " << file << ", Line: " << line << endl;
}

/*===========================================================================*/
void rabort(const char * file, int line, const char *message) {
	rpos(file, line);
	cerr << message;
	abort();
}

/*===========================================================================*/
void rexit(const char * message) {
	cout << message;
	exit(0);
}

/*###########################################################################*/
void RStd_test() {
}
}
