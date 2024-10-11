// Copyright 2017 by Robert Dick.
// All rights reserved.

#include "Graph.h"

#include <functional>
#include <algorithm>
#include <typeinfo>
#include <iostream>
#include "Epsilon.h"

namespace rstd {
using namespace std;

const RawGraph::vertex_index RawGraph::INVALID_VINDEX = -1;
const RawGraph::edge_index RawGraph::INVALID_EINDEX = -1;

/*###########################################################################*/
RawGraph &
RawGraph::operator=(const self & a) {
	vertex_ = a.vertex_;
	edge_ = a.edge_;
	return *this;
}

/*===========================================================================*/
comp_type
RawGraph::vertex_index::comp(const vertex_index & a) const {
	return rstd::comp(index_, a.index_);
}

/*===========================================================================*/
RawGraph::vertex_index RawGraph::add_vertex() {
	vertex_type new_vertex;
	vertex_.push_back(new_vertex);

	RDEBUG(self_check());
	return vertex_.size() - 1;
}

/*===========================================================================*/
comp_type
RawGraph::edge_index::comp(const edge_index & a) const {
	return rstd::comp(index_, a.index_);
}

/*===========================================================================*/
RawGraph::edge_index RawGraph::
add_edge(const vertex_index from, const vertex_index to) {
	edge_.push_back(edge_type(from, to));

	edge_index e = edge_.size() - 1;
	vertex(to)->in_.push_back(e);
	vertex(from)->out_.push_back(e);

	RDEBUG(self_check());
	return e;
}

/*===========================================================================*/
double
RawGraph::vertex_weight(vertex_index /* v */) const {
	return 0.0;
}

/*===========================================================================*/
double
RawGraph::edge_weight(edge_index /* e */) const {
	return 1.0;
}

/*===========================================================================*/
void RawGraph::erase_vertex(const vertex_index i) {
// Erase all of the connected edges.
	vertex_iterator v = vertex(i);

	while (! v->in_.empty()) {
		erase_edge(v->in_.back());
	}

	while (! v->out_.empty()) {
		erase_edge(v->out_.back());
	}

// Erase the vertex.
	vertex_.erase(v);

// Fix all of the other vertex indices.
	MAP(x, edge_.size()) {
		vertex_index & to = edge_[x].to_;
		vertex_index & from = edge_[x].from_;

		if (to > i) {
			--to;
		}

		if (from > i) {
			--from;
		}
	}
	RDEBUG(self_check());
}

/*===========================================================================*/
void RawGraph::erase_edge(const edge_index i) {
	RASSERT(i < size_edge());
// Fix connected vertices.
	vertex_type & parent = *vertex(edge(i)->from());
	vertex_type & child = *vertex(edge(i)->to());

	RVector<edge_index>::iterator parent_edge =
		find(parent.out_.begin(), parent.out_.end(), i);

	RVector<edge_index>::iterator child_edge =
		find(child.in_.begin(), child.in_.end(), i);

	parent.out_.erase(parent_edge);
	child.in_.erase(child_edge);

// Erase the edge.
	edge_.erase(edge(i));

// Fix all the other edge indices.
	MAP(x, vertex_.size()) {
		RVector<edge_index> & out = vertex_[x].out_;
		MAP(y, out.size()) {
			RASSERT(out[y] != i);
			if (out[y] > i) {
				--out[y];
			}
		}

		RVector<edge_index> & in = vertex_[x].in_;
		MAP(y, in.size()) {
			RASSERT(in[y] != i);
			if (in[y] > i) {
				--in[y];
			}
		}
	}
	RDEBUG(self_check());
}

/*===========================================================================*/
void RawGraph::pack_memory() {
	RVector<vertex_type> v(vertex_);
	RVector<edge_type> e(edge_);
	vertex_.rswap(v);
	edge_.rswap(e);
}

/*===========================================================================*/
void RawGraph::clear() {
	vertex_.clear();
	edge_.clear();
	RDEBUG(self_check());
}

/*===========================================================================*/
void RawGraph::rswap(RawGraph & rg) {
	vertex_.rswap(rg.vertex_);
	edge_.rswap(rg.edge_);
	RDEBUG(self_check());
	RDEBUG(rg.self_check());
}

/*===========================================================================*/
void RawGraph::print_to(ostream & os) const {
	MAP(x, vertex_.size()) {
		os << "vertex " << x << " ";
		vertex_[x].print_to(os);
		os << "\n";
	}

	MAP(x, edge_.size()) {
		os << "edge " << x;
		edge_[x].print_to(os);
		os << "\n";
	}
}

/*===========================================================================*/
bool RawGraph::cyclic() const {
	MAP(start, vertex_.size()) {
		RVector<big_bool> visited(vertex_.size(), false);
		if (cyclic_recurse(visited, start, start))
			return true;
	}
	return false;
}

/*===========================================================================*/
bool RawGraph::
connected(const vertex_index start, bool reverse_i) const {
	RVector<vertex_index> vec = dfs(start, reverse_i);
	return vec.size() == vertex_.size();
}

/*===========================================================================*/
bool RawGraph::
nodes_linked (vertex_index a, vertex_index b) const {
	MAP (x, vertex(a)->size_out()) {
		if (edge(vertex(a)->out(x))->to() == b) {
			return true;
		}
	}
	MAP (x, vertex(b)->size_out()) {
		if (edge(vertex(b)->out(x))->to() == a) {
			return true;
		}
	}
	return false;
}

/*===========================================================================*/
const RVector<RawGraph::vertex_index> RawGraph::
dfs(const vertex_index start, bool reverse_i) const {
	RVector<vertex_index> vec;
	RVector<big_bool> visited(vertex_.size(), false);
	dfs_recurse(vec, visited, start, reverse_i);
	return vec;
}

/*===========================================================================*/
const RVector<RawGraph::vertex_index>
RawGraph::dfs(RVector<vertex_index> start, bool reverse_i) const {
	RVector<vertex_index> vec;
	RVector<big_bool> visited(vertex_.size(), false);

	MAP(x, start.size()) {
		dfs_recurse(vec, visited, start[x], reverse_i);
	}

	return vec;
}

/*===========================================================================*/
const RVector<RawGraph::vertex_index>
RawGraph::bfs(const vertex_index start, bool reverse_i) const {
	RVector<vertex_index> vec;
	RVector<big_bool> visited(vertex_.size(), false);
	visited[start] = true;
	bfs_recurse(vec, visited, start, reverse_i);
	return vec;
}

/*===========================================================================*/
const RVector<RawGraph::vertex_index>
RawGraph::bfs(RVector<vertex_index> start, bool reverse_i) const {
	RVector<vertex_index> vec;
	RVector<big_bool> visited(vertex_.size(), false);

	MAP(x, start.size()) {
		visited[start[x]] = true;
		bfs_recurse(vec, visited, start[x], reverse_i);
	}

	return vec;
}

/*===========================================================================*/
const RVector<std::pair<RawGraph::vertex_index, double> >
RawGraph::shortest_path(vertex_index start) const {
Rabort();
	RVector<vertex_index> parent;
	RVector<double> distance;

// Initialize single source

	MAP(x, vertex_.size()) {
		parent.push_back(RawGraph::INVALID_VINDEX);

		if (x == start) {
			distance.push_back(0.0);
		} else {
			distance.push_back(numeric_limits<double>::max());
		}
	}

// relax
// recursively do this until no more changes
	bool distance_changed = true;
	while (distance_changed == true) {
		distance_changed = false;
		MAP(x, edge_.size()) {
			if (distance[edge_[x].to_] > distance[edge_[x].from_] +
				edge_weight(x))
			{
				distance[edge_[x].to_] = distance[edge_[x].from_] + edge_weight(x);
				parent[edge_[x].to_] = vertex_index(edge_[x].from_);
				distance_changed = true;
			}
		}

	}

// check for negative weight
	eps_equal_to<double, 10> eq;
	MAP(x, edge_.size()) {
//#warning THIS CODE IS WRONG
		RASSERT(eq(distance[edge_[x].to_], distance[edge_[x].from_] +
			edge_weight(x)));
	}

	RVector<std::pair<vertex_index, double> > shortestpath_result;

	MAP (x, parent.size()) {
		pair<vertex_index, double> result(parent[x], distance[x]);
		shortestpath_result.push_back(result);
	}

	return shortestpath_result;
}

/*===========================================================================*/
const RVector<RawGraph::vertex_index> RawGraph::
top_sort(const vertex_index start, bool reverse_i) const {
	RVector<vertex_index> vec;
	RVector<big_bool> visited(vertex_.size(), false);

// For now, allow only source or sink start nodes.
	RASSERT((reverse_i && ! vertex_[start].size_out()) ||
		(! reverse_i && ! vertex_[start].size_in()));

// Figure out which nodes can be reached.
	dfs_recurse(vec, visited, start, reverse_i);

// Invert to pre-visit nodes which can't be reached.
	transform(visited.begin(), visited.end(), visited.begin(),
	  logical_not<big_bool>());

	vec.erase(vec.begin(), vec.end());

	top_sort_recurse(vec, visited, start, reverse_i);
	return vec;
}

/*===========================================================================*/
const RVector<RawGraph::vertex_index> RawGraph::
top_sort(const RVector<vertex_index> & start, bool reverse_i) const {
	RVector<vertex_index> vec;
	RVector<big_bool> visited(vertex_.size(), false);

// Figure out which nodes can be reached.
	MAP(x, start.size()) {
// For now, allow only source or sink start nodes.
		RASSERT((reverse_i && ! vertex_[start[x]].size_out()) ||
			(! reverse_i && ! vertex_[start[x]].size_in()));

		dfs_recurse(vec, visited, start[x], reverse_i);
	}

// Invert to pre-visit nodes which can't be reached.
	transform(visited.begin(), visited.end(), visited.begin(),
	  logical_not<big_bool>());
	vec.erase(vec.begin(), vec.end());

	MAP(x, start.size())
		top_sort_recurse(vec, visited, start[x], reverse_i);

	return vec;
}

/*===========================================================================*/
const RVector<RawGraph::vertex_index> RawGraph::
RawGraph::outward_crawl(const vertex_index start) const {
	RVector<vertex_index> vec;
	RVector<big_bool> visited(vertex_.size(), false);
	visited[start] = true;
	outward_crawl_recurse(vec, visited, start);
	return vec;
}

/*===========================================================================*/
const RVector<int>
RawGraph::max_depth(const vertex_index start, bool reverse_i) const {
	RVector<int> vec(vertex_.size(), -1);
	RVector<big_bool> visited(vertex_.size(), false);

// Figure out which nodes can be reached.
	RVector<vertex_index> scratch;
	dfs_recurse(scratch, visited, start, reverse_i);
	scratch.erase(scratch.begin(), scratch.end());

// Invert to pre-visit nodes which can't be reached.
	transform(visited.begin(), visited.end(), visited.begin(),
	  logical_not<big_bool>());

	vec[start] = 0;
	max_depth_recurse(vec, visited, start, reverse_i);
	return vec;
}

/*===========================================================================*/
const RVector<int>
RawGraph::max_depth(const RVector<vertex_index> & start,
bool reverse_i) const {
	RVector<int> vec(vertex_.size(), -1);
	RVector<big_bool> visited(vertex_.size(), false);

	RVector<vertex_index> scratch;

	MAP(x, start.size())
		dfs_recurse(scratch, visited, start[x], reverse_i);

	scratch.erase(scratch.begin(), scratch.end());

	transform(visited.begin(), visited.end(), visited.begin(),
	  logical_not<big_bool>());

	MAP(x, start.size()) {
		vec[start[x]] = 0;
		max_depth_recurse(vec, visited, start[x], reverse_i);
	}

	return vec;
}

/*===========================================================================*/
void RawGraph::self_check() const {
	MAP(x, vertex_.size()) {
		Rassert(vertex_offset(vertex_.begin() + x) < size_vertex());
	}

	MAP(x, edge_.size()) {
		Rassert(edge_[x].to_ < size_vertex());
		Rassert(edge_[x].from_ < size_vertex());
	}

	MAP(x, vertex_.size()) {
		MAP(y, vertex_[x].out_.size()) {
			Rassert(vertex_[x].out_[y] < size_edge());
		}

		MAP(y, vertex_[x].in_.size()) {
			Rassert(vertex_[x].in_[y] < size_edge());
		}
	}
}

/*===========================================================================*/
bool RawGraph::cyclic_recurse(RVector<big_bool> & visited,
const vertex_index start, const vertex_index branch) const {
	visited[branch] = true;
	MAP(x, vertex(branch)->out_.size()) {
		vertex_index to_vertex = edge(vertex(branch)->out_[x])->to_;

		if (to_vertex == start)
			return true;

		if (! visited[to_vertex])
			if (cyclic_recurse(visited, start, to_vertex))
				return true;
	}
	return false;
}

/*===========================================================================*/
void RawGraph::dfs_recurse(RVector<vertex_index> & vec,
RVector<big_bool> & visited, const vertex_index branch,
bool reverse_i) const {
	visited[branch] = true;
	vec.push_back(branch);

	if (reverse_i) {
		MAP(x, vertex(branch)->in_.size()) {
			vertex_index next_index = edge(vertex(branch)->in_[x])->from_;
			if (! visited[next_index]) {
				dfs_recurse(vec, visited, next_index, reverse_i);
			}
		}
	} else {
		MAP(x, vertex(branch)->out_.size()) {
			vertex_index next_index = edge(vertex(branch)->out_[x])->to_;
			if (! visited[next_index]) {
				dfs_recurse(vec, visited, next_index, reverse_i);
			}
		}
	}
}

/*===========================================================================*/
void RawGraph::bfs_recurse(RVector<vertex_index> & vec,
RVector<big_bool> & visited, const vertex_index branch,
bool reverse_i) const {
	RVector<vertex_index> stack;

	if (reverse_i) {
		MAP(x, vertex(branch)->in_.size()) {
			vertex_index next_index = edge(vertex(branch)->in_[x])->from_;

			if (! visited[next_index]){
				visited[next_index] = true;
				vec.push_back(next_index);
				stack.push_back(next_index);
			}
		}
	} else {
		MAP(x, vertex(branch)->out_.size()) {
			vertex_index next_index = edge(vertex(branch)->out_[x])->to_;

			if (! visited[next_index]){
				visited[next_index] = true;
				vec.push_back(next_index);
				stack.push_back(next_index);
			}
		}
	}

	MAP(x, stack.size()) {
		bfs_recurse(vec, visited, stack[x], reverse_i);
	}
}

/*===========================================================================*/
void RawGraph::top_sort_recurse(RVector<vertex_index> & vec,
RVector<big_bool> & visited, const vertex_index branch,
bool reverse_i) const {
// If any of the parents haven't been visited yet, stop exploring this path.
	if (reverse_i) {
		MAP(x, vertex(branch)->out_.size())
			if (! visited[edge(vertex(branch)->out_[x])->to_])
				return;
	} else {
		MAP(x, vertex(branch)->in_.size())
			if (! visited[edge(vertex(branch)->in_[x])->from_])
				return;
	}

	visited[branch] = true;
	vec.push_back(branch);

	if (reverse_i) {
		MAP(x, vertex(branch)->in_.size()) {
			vertex_index next_index = edge(vertex(branch)->in_[x])->from_;
			if (! visited[next_index])
				top_sort_recurse(vec, visited, next_index, reverse_i);
		}
	} else {
		MAP(x, vertex(branch)->out_.size()) {
			vertex_index next_index = edge(vertex(branch)->out_[x])->to_;
			if (! visited[next_index])
				top_sort_recurse(vec, visited, next_index, reverse_i);
		}
	}
}

/*===========================================================================*/
void RawGraph::outward_crawl_recurse(RVector<vertex_index> & vec,
RVector<big_bool> & visited, vertex_index branch) const {
	RVector<vertex_index> stack;

	MAP(x, vertex(branch)->in_.size()) {
		vertex_index next_index = edge(vertex(branch)->in_[x])->from_;

		if (! visited[next_index]){
			visited[next_index] = true;
			vec.push_back(next_index);
			stack.push_back(next_index);
		}
	}

	MAP(x, vertex(branch)->out_.size()) {
		vertex_index next_index = edge(vertex(branch)->out_[x])->to_;

		if (! visited[next_index]){
			visited[next_index] = true;
			vec.push_back(next_index);
			stack.push_back(next_index);
		}
	}

	MAP(x, stack.size()) {
		outward_crawl_recurse(vec, visited, stack[x]);
	}
}

/*===========================================================================*/
void RawGraph::max_depth_recurse(RVector<int> & vec,
RVector<big_bool> & visited, const vertex_index branch,
bool reverse_i) const {
// If any of the parents haven't been visited yet, stop exploring this path.
	if (reverse_i) {
		MAP(x, vertex(branch)->out_.size())
			if (! visited[edge(vertex(branch)->out_[x])->to_])
				return;
	} else {
		MAP(x, vertex(branch)->in_.size())
			if (! visited[edge(vertex(branch)->in_[x])->from_])
				return;
	}

	visited[branch] = true;

	if (reverse_i) {
		MAP(x, vertex(branch)->in_.size()) {
			vertex_index next_index = edge(vertex(branch)->in_[x])->from_;
			vec[next_index] = max(vec[next_index], vec[branch] + 1);
			if (! visited[next_index])
				max_depth_recurse(vec, visited, next_index, reverse_i);
		}
	} else {
		MAP(x, vertex(branch)->out_.size()) {
			vertex_index next_index = edge(vertex(branch)->out_[x])->to_;
			vec[next_index] = max(vec[next_index], vec[branch] + 1);
			if (! visited[next_index])
				max_depth_recurse(vec, visited, next_index, reverse_i);
		}
	}
}

/*===========================================================================*/
void RawGraph::edge_type::print_to(ostream & os) const {
	os << " from vertex " << from_ << " to vertex " << to_;
}

/*===========================================================================*/
comp_type
RawGraph::edge_type::comp(const self & et) const {
	const comp_type fr = rstd::comp(from_, et.from_);
	return fr ? fr : rstd::comp(to_, et.to_);
}

/*===========================================================================*/
void RawGraph::vertex_type::print_to(ostream & os) const {
	os << "out edges: ";
	MAP(y, out_.size()) {
		os << out_[y] << " ";
	}

	os << "in edges: ";
	MAP(y, in_.size()) {
		os << in_[y] << " ";
	}
}

/*===========================================================================*/
comp_type
RawGraph::vertex_type::comp(const self & vt) const {
	const comp_type ot = comp_cont(out_, vt.out_);
	return ot ? ot : comp_cont(in_, vt.in_);
}

/*###########################################################################*/
double
WGraph::vertex_weight(vertex_index v) const {
	return (*this)[v];
}

/*===========================================================================*/
double
WGraph::edge_weight(edge_index e) const {
	return (*this)(e);
}

/*###########################################################################*/
void Graph_test() {
	Graph<int, int> g1;
	Graph<int, int> g2;

	g1.rswap(g2);

	typedef RawGraph::vertex_index vi;
	vi v0 = g1.add_vertex(0);
	vi v1 = g1.add_vertex(1);
	vi v2 = g1.add_vertex(2);
	vi v3 = g1.add_vertex(3);

	g1.add_edge(v0, v1, 0);
	g1.add_edge(v1, v2, 1);
	g1.add_edge(v0, v3, 2);

	RVector<RawGraph::vertex_index> dl;
	dl.push_back(v3);
	dl.push_back(v2);
	dl.push_back(v1);

	for (long x = dl.size() - 1; x >= 0; --x) {
		if (g1.vertex(dl[x])->size_out()) {
			dl.erase(dl.begin() + x);
		}
	}

	cout << g1 << "\n";

	RVector<RawGraph::vertex_index> rtsrt = g1.top_sort(dl, true);
	cout << "Graph: " << rtsrt << "\n";
	RASSERT(rtsrt.size() == g1.size_vertex());
}

}
