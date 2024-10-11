// Copyright 2017 by Robert Dick.
// All rights reserved.

#ifndef GRAPH_H_
#define GRAPH_H_

/*###########################################################################*/
#include "RStd.h"
#include "RVector.h"
#include "Interface.h"

#ifdef ROB_DEBUG
#	include <typeinfo>
#endif

#include <iosfwd>
#include <cstddef>

namespace rstd {
/*###########################################################################*/
// Base data-independent directed graph class.

class RawGraph :
	public Prints<RawGraph>,
	public Clones<RawGraph>,
	public Swaps<RawGraph>,
	public SChecks<RawGraph>
{
	typedef RawGraph self;

public:
// Type safe indexes.

	class vertex_index :
		public Prints<vertex_index>,
		public Comps<vertex_index>
	{
	public:
		vertex_index(long indx) : index_(indx) {}
		operator long() const { return index_; }
		comp_type comp(const vertex_index & a) const;

		vertex_index & operator++() { index_++; return *this; }
		vertex_index operator++(int);
		vertex_index & operator--() { index_--; return *this; }
		vertex_index operator--(int);
		vertex_index operator+=(vertex_index i);
		vertex_index operator-=(vertex_index i);
		void print_to(std::ostream & os) const { os << index_; }

	private:
			long index_;
	};

	class edge_index :
		public Prints<edge_index>,
		public Comps<edge_index>
	{
	public:
		edge_index(long indx) : index_(indx) {}
		operator long() const { return index_; }

		comp_type comp(const edge_index & a) const;

		edge_index & operator++() { index_++; return *this; }
		edge_index operator++(int);
		edge_index & operator--() { index_--; return *this; }
		edge_index operator--(int);
		edge_index operator+=(edge_index i);
		edge_index operator-=(edge_index i);
		void print_to(std::ostream & os) const { os << index_; }

	private:
			long index_;
	};

	class vertex_type :
		public Prints<RawGraph::vertex_type>
	{
		typedef RawGraph::vertex_type self;

	public:
		typedef RVector<edge_index>::iterator edge_iterator;
		typedef RVector<edge_index>::const_iterator const_edge_iterator;

		vertex_type() : out_(), in_() {}
		edge_index size_out() const { return out_.size(); }
		edge_index out(long i) const { return out_[i]; }
		edge_index size_in() const { return in_.size(); }
		edge_index in(long i) const { return in_[i]; }

		const_edge_iterator begin_out() const { return out_.begin(); }
		edge_iterator begin_out() { return out_.begin(); }
		const_edge_iterator end_out() const { return out_.end(); }
		edge_iterator end_out() { return out_.end(); }

		const_edge_iterator begin_in() const { return in_.begin(); }
		edge_iterator begin_in() { return in_.begin(); }
		const_edge_iterator end_in() const { return in_.end(); }
		edge_iterator end_in() { return in_.end(); }

		void print_to(std::ostream & os) const;
		comp_type comp(const self & a) const;

	protected:
			RVector<edge_index> out_;
			RVector<edge_index> in_;

		friend class rstd::RawGraph;
	};

	class edge_type :
		public Prints<RawGraph::edge_type>,
		public Comps<RawGraph::edge_type>
	{
		typedef RawGraph::edge_type self;

	public:
		vertex_index from() const { return from_; }
		vertex_index to() const { return to_; }

		void print_to(std::ostream & os) const;

		comp_type comp(const self & et) const;

	protected:
		edge_type(vertex_index from, vertex_index to);

			vertex_index from_;
			vertex_index to_;

		friend class rstd::RawGraph;
	};

private:
	typedef RVector<vertex_type> v_impl;
	typedef RVector<edge_type> e_impl;

public:
// Typedefs
	typedef ptrdiff_t difference_type;

	typedef v_impl::iterator vertex_iterator;
	typedef e_impl::iterator edge_iterator;
	typedef v_impl::const_iterator const_vertex_iterator;
	typedef e_impl::const_iterator const_edge_iterator;

	typedef v_impl::reverse_iterator reverse_vertex_iterator;
	typedef e_impl::reverse_iterator reverse_edge_iterator;
	typedef v_impl::const_reverse_iterator const_reverse_vertex_iterator;
	typedef e_impl::const_reverse_iterator const_reverse_edge_iterator;

	typedef v_impl::reference vertex_reference;
	typedef e_impl::reference edge_reference;
	typedef v_impl::const_reference const_vertex_reference;
	typedef e_impl::const_reference const_edge_reference;

// Construction
	virtual ~RawGraph() {}
	RawGraph() : vertex_(), edge_() {}
	virtual self & operator=(const self & a);

// Interface
	virtual void rswap(self & rg);
	virtual self * clone() const { return new self(*this); }
	virtual void print_to(std::ostream & os) const;
	virtual void self_check() const;
	virtual void self_check_deep() const { self_check(); }

// Modifiable
// Invalidates indices.  This is slow, O(v * e).
	virtual void erase_vertex(vertex_index i);

// Invalidates indices.  This is slow, O(v * e).
	virtual void erase_edge(edge_index i);

// Tries to eliminate any padding memory.  Run when size fixed.
	virtual void pack_memory();
	virtual void clear();

// Final
	vertex_index size_vertex() const;
	edge_index size_edge() const;
	vertex_index vertex_offset(const_vertex_iterator x) const;
	edge_index edge_offset(const_edge_iterator x) const;
	vertex_index vertex_offset(const_reverse_vertex_iterator x) const;
	edge_index edge_offset(const_reverse_edge_iterator x) const;
	vertex_index add_vertex();
	edge_index add_edge(vertex_index from, vertex_index to);

	virtual double vertex_weight(vertex_index v) const;
	virtual double edge_weight(edge_index e) const;

	edge_iterator edge_begin() { return edge_.begin(); }
	const_edge_iterator edge_begin() const { return edge_.begin(); }
	edge_iterator edge_end() { return edge_.end(); }
	const_edge_iterator edge_end() const { return edge_.end(); }

	reverse_edge_iterator edge_rbegin() { return edge_.rbegin(); }
	const_reverse_edge_iterator edge_rbegin() const { return edge_.rbegin(); }
	reverse_edge_iterator edge_rend() { return edge_.rend(); }
	const_reverse_edge_iterator edge_rend() const { return edge_.rend(); }

	vertex_iterator vertex_begin() { return vertex_.begin(); }
	const_vertex_iterator vertex_begin() const { return vertex_.begin(); }
	vertex_iterator vertex_end() { return vertex_.end(); }
	const_vertex_iterator vertex_end() const { return vertex_.end(); }

	reverse_vertex_iterator vertex_rbegin() { return vertex_.rbegin(); }
	const_reverse_vertex_iterator vertex_rbegin() const;
	reverse_vertex_iterator vertex_rend() { return vertex_.rend(); }
	const_reverse_vertex_iterator vertex_rend() const { return vertex_.rend(); }

	vertex_iterator vertex(vertex_index index);
	const_vertex_iterator vertex(vertex_index index) const;
	edge_iterator edge(edge_index index);
	const_edge_iterator edge(edge_index index) const;

// Checks for cycles in the graph.
	bool cyclic() const;

// Confirms that all vertices are connected to start.
	bool connected(vertex_index start, bool reverse = false) const;

// Checks if an edge exists between 2 nodes.
	bool nodes_linked(vertex_index a, vertex_index b) const;

// Returns a DFS-ordered RVector of vertex indices.
	const RVector<vertex_index>
		dfs(vertex_index start, bool reverse = false) const;

	const RVector<vertex_index>
		dfs(RVector<vertex_index> start, bool reverse = false) const;

// Returns a BFS-ordered RVector of vertex indices.
	const RVector<vertex_index>
		bfs(vertex_index start, bool reverse = false) const;

	const RVector<vertex_index>
		bfs(RVector<vertex_index> start, bool reverse = false) const;

// Return every vertex's parent vertex and shortest path distance
	const RVector<std::pair<vertex_index, double> >
		shortest_path(vertex_index start) const;

// Returns a topological sort-ordered RVector of vertex indices.
	const RVector<vertex_index> top_sort(vertex_index start,
		bool reverse = false) const;

	const RVector<vertex_index> top_sort(const RVector<vertex_index> & start,
		bool reverse = false) const;

// BFS which ignores edge directions.
	const RVector<vertex_index>
		outward_crawl(vertex_index start) const;

// Disconnected vertices will have depths less than 0.
	const RVector<int> max_depth(vertex_index start,
		bool reverse = false) const;

	const RVector<int> max_depth(const RVector<vertex_index> & start,
		bool reverse = false) const;

		static const vertex_index INVALID_VINDEX;
		static const edge_index INVALID_EINDEX;

protected:
	bool cyclic_recurse(RVector<big_bool> & visited,
		vertex_index start, vertex_index branch) const;

	void dfs_recurse(RVector<vertex_index> & vec, RVector<big_bool> & visited,
		vertex_index branch, bool reverse) const;

	void bfs_recurse(RVector<vertex_index> & vec, RVector<big_bool> & visited,
		vertex_index branch, bool reverse) const;

	void top_sort_recurse(RVector<vertex_index> & vec,
		RVector<big_bool> & visited, vertex_index branch,
		bool reverse) const;

	void outward_crawl_recurse(RVector<vertex_index> & vec,
		RVector<big_bool> & visited, vertex_index branch) const;

	void max_depth_recurse(RVector<int> & vec,
		RVector<big_bool> & visited, vertex_index branch,
		bool reverse) const;

private:
		v_impl vertex_;
		e_impl edge_;
};

RawGraph::vertex_index operator+(RawGraph::vertex_index a,
	RawGraph::vertex_index b);

RawGraph::vertex_index operator-(RawGraph::vertex_index a,
	RawGraph::vertex_index b);

RawGraph::edge_index operator+(RawGraph::edge_index a,
	RawGraph::edge_index b);

RawGraph::edge_index operator-(RawGraph::edge_index a,
	RawGraph::edge_index b);

/*===========================================================================*/
// Type-specific directed graph.

template <typename V, typename E>
class Graph :
	public RawGraph
{
	typedef RawGraph super;
	typedef Graph self;

public:
	Graph() : super(), v_data_(), e_data_() {};

// Interface
	virtual void rswap(RawGraph & g);
	virtual self * clone() const { return new self(*this); }
	virtual void self_check_deep() const;

// Modifiable
	virtual vertex_index add_vertex(const V & data);

	virtual edge_index add_edge(vertex_index from,
		vertex_index to, const E & data);

	virtual void erase_vertex(vertex_index i);
	virtual void erase_edge(edge_index i);

	virtual void pack_memory();
	virtual void clear();

// Needs a graph label
	virtual void print_to_vcg(std::ostream & os, const std::string & glab) const;

// Final
	void print_to_default(std::ostream & os) const;

//	void set_v_data(vertex_index i, const V & data) const {v_data_[i](data);
//			return;}

	V & operator[](vertex_index i) { return v_data_[i]; }
	const V & operator[](vertex_index i) const { return v_data_[i]; }

	E & operator()(edge_index i) { return e_data_[i]; }
	const E & operator()(edge_index i) const { return e_data_[i]; }

private:
		RVector<V> v_data_;
		RVector<E> e_data_;
};

/*===========================================================================*/
class WGraph :	public Graph<double, double> {
public:
	virtual double vertex_weight(vertex_index v) const;
	virtual double edge_weight(edge_index e) const;
};

/*===========================================================================*/
void Graph_test();

/*###########################################################################*/
#include "Graph.cct"
}
#endif
