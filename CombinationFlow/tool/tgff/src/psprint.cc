// Copyright 2017 by Robert Dick, David Rhodes, and Keith Vallerio.
// All rights reserved.

#include <cstdlib>
#include <cmath>
#include <string>
#include <iostream>
#include <cmath>
#include <climits>

#include "RStd.h"
#include "ArgPack.h"
#include "TGraph.h"
#include "PGraph.h"

using namespace std;
using namespace rstd;

/*###########################################################################*/
static const int PSxoff = 50;
static const int PSyoff = 50;
static const int PSwidth = 500;
static const int PSheight = 628;

static int	X_min, Y_min,	// actual x/y mins/max used (for BB)
			X_max, Y_max;

/*===========================================================================*/
static void i_PS_dag     (ostream&, const RVector<TG> &dag, int);
static void i_PS_node    (ostream&, const RVector<TG> &dag, int, int, int);
static void i_PS_arc     (ostream&, const RVector<TG> &dag, int, int, int);
static void i_PS_deadline(ostream&, const RVector<TG> &dag, int, int, int);
static void i_PS_locnode(int, int, int, int, int,
					int *, int *, int *, int *, int, int);
inline static int PS_x_out(int);
inline static int PS_y_out(int);

/*###########################################################################*/
void TGraph::print_to_ps(ostream & os, const string & filename) const {
	RASSERT(dag_.size());

	X_min = Y_min = INT_MAX;
	X_max = Y_max = 0;

	// write header
	os << "%!PS-Adobe-2.0\n%%Creator: TGFF by R. Dick and D. Rhodes\n";
	os << "%%Pages: 1\n%%Title: " << filename << "\n";
	os << "0.2 setlinewidth\n";

	int		fmaxw, fmaxh;
	fmaxw = fmaxh = -1;
	MAP(x, dag_.size()) {
		fmaxw = max(fmaxw, dag_[x].max_w());
		fmaxh = max(fmaxh, dag_[x].max_h());
	}

	// set font and scale
	const double min_PS   = static_cast<double> (min(PSheight, PSwidth));
	const double box_size = min_PS /
				(1.0 + max(1, max(static_cast<int> (dag_.size() * fmaxw),
								  fmaxh)));
	int fsize = static_cast<int> ( box_size / 4.0 );
	os << "/Courier findfont " << fsize << " scalefont setfont\n";

	// DRAW OBJECTS ...
	i_PS_dag     (os, dag_, fsize);
	i_PS_node    (os, dag_, fmaxw, fmaxh, fsize);
	i_PS_arc     (os, dag_, fmaxw, fmaxh, fsize);
	i_PS_deadline(os, dag_, fmaxw, fmaxh, fsize);

	int pad = max(1,2*fsize);
	X_min -= pad;
	Y_min -= pad;
	X_max += pad;
	Y_max += pad;

	// Put in BB
	os	<< "%%BoundingBox: "
			<< X_min << " " << Y_min << " " << X_max << " " << Y_max << endl;
	os << "showpage" << endl;
}
/*###########################################################################*/
void PGraph::print_to_ps(ostream & os, const string & filename) const {
	// ALAS: identical to TGraph version 
	RASSERT(dag_.size());

	X_min = Y_min = 10000;
	X_max = Y_max = 0;

	// write header
	os << "%!PS-Adobe-2.0\n%%Creator: TGFF by R. Dick and D. Rhodes\n";
	os << "%%Pages: 1\n%%Title: " << filename << "\n";
	os << "0.2 setlinewidth\n";

	int		fmaxw, fmaxh;
	fmaxw = fmaxh = -1;
	MAP(x, dag_.size()) {
		fmaxw = max(fmaxw, dag_[x].max_w());
		fmaxh = max(fmaxh, dag_[x].max_h());
	}

	// set font and scale
	const double min_PS   = static_cast<double> (min(PSheight, PSwidth));
	const double box_size = min_PS /
				(1.0 + max(1, max(static_cast<int> (dag_.size() * fmaxw),
								  fmaxh)));
	int fsize = static_cast<int> ( box_size / 4.0 );
	os << "/Courier findfont " << fsize << " scalefont setfont\n";

	// DRAW OBJECTS ...
	i_PS_dag     (os, dag_, fsize);
	i_PS_node    (os, dag_, fmaxw, fmaxh, fsize);
	i_PS_arc     (os, dag_, fmaxw, fmaxh, fsize);
	i_PS_deadline(os, dag_, fmaxw, fmaxh, fsize);

	int pad = max(1,2*fsize);
	X_min -= pad;
	Y_min -= pad;
	X_max += pad;
	Y_max += pad;

	// Put in BB
	os	<< "%%BoundingBox: "
			<< X_min << " " << Y_min << " " << X_max << " " << Y_max << endl;
	os << "showpage" << endl;
}

// ---------------------------------------------------------------------------

inline static int PS_x_out(int x) {
	int v = PSxoff + x;
	X_min = min(X_min,v);
	X_max = max(X_max,v);
	return (v);
}
	
// ---------------------------------------------------------------------------

inline static int PS_y_out(int y) {
// flip
	int v = PSxoff + PSheight - y;
	Y_min = min(Y_min,v);
	Y_max = max(Y_max,v);
	return (v);
}

// ---------------------------------------------------------------------------

static void i_PS_locnode(int maxw, int maxh, int fsize, int indx,
int numx, int *x1, int *x2, int *yy1, int *yy2, int h, int w) {
	RASSERT(numx > 0);
	double PS_pert_size = 0.4;
	double min_PS   = static_cast<double> (min(PSheight,PSwidth));
	double box_size = min_PS /
					(1.0 + static_cast<double> (max(1, max(numx*maxw,maxh))));
	double taskoffset = indx * (min_PS / numx);

	// Let x position vary slightly with vertical to get some skewing
	// to lessen inadequacy of rat's nest drawing.
	//
	// SIMPLEST, JUST DO: double xadjust = 0.0;

	double phi = 6.28 * static_cast<double> (h) /
									static_cast<double> (maxh);
	double xadjust = 0.75+cos(phi)/2.0;

	double xp = taskoffset + box_size*(w + xadjust);
	double yp = 4*fsize    + box_size*(h + 0.5);
	double xyr = box_size * PS_pert_size/2.0;

	*x1 = static_cast<int> (xp - xyr);
	*x2 = static_cast<int> (xp + xyr);
	*yy1 = static_cast<int> (yp - xyr);
	*yy2 = static_cast<int> (yp + xyr);
}

// ---------------------------------------------------------------------------

static void i_PS_dag(ostream& os, const RVector<TG> &dag, int fsize) {
	MAP(indx, dag.size()) {		// for each dag
		double min_PS   = static_cast<double> (min(PSheight,PSwidth));
		double taskoffset = static_cast<double>(indx) * (min_PS /
			static_cast<double>(dag.size()));

		int	x1, yy1;
		x1 = static_cast<int> (taskoffset);
		yy1 = 2*fsize;
		os	<< "newpath " << PS_x_out(x1) << " " << PS_y_out(yy1) << " moveto "
			<< "(TASK_GRAPH " << indx << ") show\n";
		yy1 += fsize;
		os	<< "newpath " << PS_x_out(x1) << " " << PS_y_out(yy1) << " moveto "
			<< "(  Period= " << dag[indx].period() << ") show\n";
		yy1 += fsize;
		os	<< "newpath " << PS_x_out(x1) << " " << PS_y_out(yy1) << " moveto "
			<< "(  In/Out Degree Limits= " <<
				dag[indx].in_deg() << " / " << dag[indx].out_deg() << ") show\n";
	}
}

// ---------------------------------------------------------------------------

static void i_PS_node(ostream& os, const RVector<TG> &dag, int fmaxw,
int fmaxh, int fsize) {
	int numx = static_cast<int>(dag.size());
	RASSERT(numx > 0);

	MAPI (indx, dag.size()) {
		MAPI (x, dag[indx].size_vertex()) {
			int	x1, x2, yy1, yy2;
			i_PS_locnode(fmaxw, fmaxh, fsize, indx, numx,
				&x1, &x2, &yy1, &yy2, dag[indx][x].h,
				dag[indx][x].w);
			int y1bump = yy1 + static_cast<int> (0.75*(yy2-yy1));

			// flip y-coords
			// not best way to do it
			os
			<< "newpath "
			<< PS_x_out(x1) << " " << PS_y_out(yy1) << " moveto "
			<< PS_x_out(x1) << " " << PS_y_out(yy2) << " lineto stroke\n"
			<< "newpath "
			<< PS_x_out(x1) << " " << PS_y_out(yy2) << " moveto "
			<< PS_x_out(x2) << " " << PS_y_out(yy2) << " lineto stroke\n"
			<< "newpath "
			<< PS_x_out(x2) << " " << PS_y_out(yy2) << " moveto "
			<< PS_x_out(x2) << " " << PS_y_out(yy1) << " lineto stroke\n"
			<< "newpath "
			<< PS_x_out(x2) << " " << PS_y_out(yy1) << " moveto "
			<< PS_x_out(x1) << " " << PS_y_out(yy1) << " lineto stroke\n";
			// label
			os	<< "newpath "
			<< PS_x_out(x1) << " " << PS_y_out(y1bump) << " moveto "
									<< "(" << dag[indx][x].name << ") show\n";
		}
	}
}

/*===========================================================================*/
static void i_PS_arc(ostream& os, const RVector<TG> &dag, int fmaxw,
int fmaxh, int fsize) {
	int numx = static_cast<int>(dag.size());
	RASSERT(numx > 0);

	MAPI (indx, dag.size()) {
		MAPI (x, dag[indx].size_edge()) {
			int n1 = static_cast<int>(dag[indx].edge(x)->from());
			int n2 = static_cast<int>(dag[indx].edge(x)->to());

			int	x11,x12,y11,y12;
			int	x21,x22,y21,y22;
			i_PS_locnode(fmaxw, fmaxh, fsize, indx, numx,
						 &x11, &x12, &y11, &y12, dag[indx][n1].h, dag[indx][n1].w);
			i_PS_locnode(fmaxw, fmaxh, fsize, indx, numx,
						 &x21, &x22, &y21, &y22, dag[indx][n2].h, dag[indx][n2].w);

			int mx1 = (x11+x12)/2;
			int my1 = y12;
			int mx2 = (x21+x22)/2;
			int my2 = y21;

			os << "newpath "
			<< PS_x_out(mx1) << " " << PS_y_out(my1) << " moveto "
			<< PS_x_out(mx2) << " " << PS_y_out(my2) << " lineto stroke\n";
			// put arrow on
			const double arrow_fan = 0.15;				// 15% fan
			const double arrow_len = 0.25 * (x22-x21);	// 25% of box size
			double deltax = (mx2-mx1);
			double deltay = (my2-my1);

			RASSERT((deltax*deltax + deltay*deltay) >= 0.0);
			double per = arrow_len / sqrt(deltax*deltax + deltay*deltay);
			double dax = deltax * per;
			double day = deltay * per;
			double xline = mx2 - dax;
			double yline = my2 - day;

			double daax = dax * arrow_fan;
			double daay = day * arrow_fan;
			int a1x = static_cast<int> (xline-daay);
			int a1y = static_cast<int> (yline+daax);
			os << "newpath "
				<< PS_x_out(mx2) << " " << PS_y_out(my2) << " moveto "
				<< PS_x_out(a1x) << " " << PS_y_out(a1y) << " lineto stroke\n";

			a1x = static_cast<int> (xline+daay);
			a1y = static_cast<int> (yline-daax);
			os << "newpath "
				<< PS_x_out(mx2) << " " << PS_y_out(my2) << " moveto "
				<< PS_x_out(a1x) << " " << PS_y_out(a1y) << " lineto stroke\n";
		}
	}
}

// ---------------------------------------------------------------------------

static void i_PS_deadline(ostream& os, const RVector<TG> &dag, int fmaxw,
int fmaxh, int fsize) {
	int numx = static_cast<int>(dag.size());
	RASSERT(numx > 0);

	MAPI (indx, dag.size()) {
		MAPI (x, dag[indx].size_vertex()) {
			if( dag[indx][x].deadline <= 0 ) continue;

			int	x1,x2,yy1,yy2;
			i_PS_locnode(fmaxw, fmaxh, fsize, indx, numx,
						 &x1, &x2, &yy1, &yy2, dag[indx][x].h, dag[indx][x].w);
			int y1bump = yy2 + static_cast<int> (0.75*(yy2-yy1));

			// label
			os	<< "newpath "
					<< PS_x_out(x1) << " " << PS_y_out(y1bump) << " moveto "
							<< "(" << "d=" << dag[indx][x].deadline << ") show\n";
		}
	}
}






