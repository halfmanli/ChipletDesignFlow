// $Id$

/*
 Copyright (c) 2007-2015, Trustees of The Leland Stanford Junior University
 All rights reserved.

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:

 Redistributions of source code must retain the above copyright notice, this 
 list of conditions and the following disclaimer.
 Redistributions in binary form must reproduce the above copyright notice, this
 list of conditions and the following disclaimer in the documentation and/or
 other materials provided with the distribution.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE 
 DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
 ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/*anynet
 *
 *Network setup file format
 *example 1:
 *router 0 router 1 15 router 2
 *
 *Router 0 is connect to router 1 with a 15-cycle channel, and router 0 is connected to
 * router 2 with a 1-cycle channel, the channels latency are unidirectional, so channel 
 * from router 1 back to router 0 is only single-cycle because it was not specified
 *
 *example 2:
 *router 0 node 0 node 1 5 node 2 5
 *
 *Router 0 is directly connected to node 0-2. Channel latency is 5cycles for 1 and 2. In 
 * this case the latency specification is bidirectional, the injeciton and ejection lat
 * for node 1 and 2 are 5-cycle
 *
 *other notes:
 *
 *Router and node numbers must be sequential starting with 0
 *Credit channel latency follows the channel latency, even though it travels in revse
 * direction this might not be desired
 *
 */

#include "anynet.hpp"
#include <fstream>
#include <sstream>
#include <limits>
#include <algorithm>

//this is a hack, I can't easily get the routing talbe out of the network
// map<int, int> *global_routing_table;  // map<dst, port>
map<pair<int, int>, vector<int>> *global_routing_table;  // map<(src, dst), <port_hop_0, port_hop_1, ...>>

AnyNet::AnyNet(const Configuration &config, const string &name)
        : Network(config, name)
{

    router_list.resize(2);
    _ComputeSize(config);
    _Alloc();
    _BuildNet(config);
}

AnyNet::~AnyNet()
{
    for (int i = 0; i < 2; ++i)
    {
        for (auto iter = router_list[i].begin();
             iter != router_list[i].end();
             ++iter)
        {
            iter->second.clear();
        }
    }
}

void AnyNet::_ComputeSize(const Configuration &config)
{
    file_name = config.GetStr("network_file");
    if (file_name == "")
    {
        cout << "No network file name provided" << endl;
        exit(-1);
    }
    //parse the network description file
    readFile();

    _channels = 0;
    cout << "========================Network File Parsed=================\n";
    cout << "******************node listing**********************\n";
    map<int, int>::iterator iter;
    for (iter = node_list.begin(); iter != node_list.end(); iter++)
    {
        cout << "Node " << iter->first;
        cout << "\tRouter " << iter->second << endl;
    }

    map<int, map<int, pair<int, int> > >::iterator iter3;
    cout << "\n****************router to node listing*************\n";
    for (iter3 = router_list[0].begin(); iter3 != router_list[0].end(); iter3++)
    {
        cout << "Router " << iter3->first << endl;
        map<int, pair<int, int> >::iterator iter2;
        for (iter2 = iter3->second.begin();
             iter2 != iter3->second.end();
             iter2++)
        {
            cout << "\t Node " << iter2->first << " lat " << iter2->second.second << endl;
        }
    }

    cout << "\n*****************router to router listing************\n";
    for (iter3 = router_list[1].begin(); iter3 != router_list[1].end(); iter3++)
    {
        cout << "Router " << iter3->first << endl;
        map<int, pair<int, int> >::iterator iter2;
        if (iter3->second.size() == 0)
        {
            cout << "Caution Router " << iter3->first
                 << " is not connected to any other Router\n" << endl;
        }
        for (iter2 = iter3->second.begin();
             iter2 != iter3->second.end();
             iter2++)
        {
            cout << "\t Router " << iter2->first << " lat " << iter2->second.second << endl;
            _channels++;
        }
    }

    _size = router_list[1].size();
    _nodes = node_list.size();

}


void AnyNet::_BuildNet(const Configuration &config)
{


    //I need to keep track the output ports for each router during build
    int *outport = (int *) malloc(sizeof(int) * _size);
    for (int i = 0; i < _size; i++)
    { outport[i] = 0; }

    cout << "==========================Node to Router =====================\n";
    //adding the injection/ejection chanenls first
    map<int, map<int, pair<int, int> > >::iterator niter;
    for (niter = router_list[0].begin(); niter != router_list[0].end(); niter++)
    {
        map<int, map<int, pair<int, int> > >::iterator riter = router_list[1].find(niter->first);
        //calculate radix
        int radix = niter->second.size() + riter->second.size();
        int node = niter->first;
        cout << "router " << node << " radix " << radix << endl;
        //declare the routers
        ostringstream router_name;
        router_name << "router";
        router_name << "_" << node;
        _routers[node] = Router::NewRouter(config, this, router_name.str(),
                                           node, radix, radix);
        _timed_modules.push_back(_routers[node]);
        //add injeciton ejection channels
        map<int, pair<int, int> >::iterator nniter;
        for (nniter = niter->second.begin(); nniter != niter->second.end(); nniter++)
        {
            int link = nniter->first;
            //add the outport port assined to the map
            (niter->second)[link].first = outport[node];
            outport[node]++;
            cout << "\t connected to node " << link << " at outport " << nniter->second.first
                 << " lat " << nniter->second.second << endl;
            _inject[link]->SetLatency(nniter->second.second);
            _inject_cred[link]->SetLatency(nniter->second.second);
            _eject[link]->SetLatency(nniter->second.second);
            _eject_cred[link]->SetLatency(nniter->second.second);

            _routers[node]->AddInputChannel(_inject[link], _inject_cred[link]);
            _routers[node]->AddOutputChannel(_eject[link], _eject_cred[link]);
        }

    }

    cout << "==========================Router to Router =====================\n";
    //add inter router channels
    //since there is no way to systematically number the channels we just start from 0
    //the map, is a mapping of output->input
    int channel_count = 0;
    for (niter = router_list[0].begin(); niter != router_list[0].end(); niter++)
    {
        auto riter = router_list[1].find(niter->first);
        int node = niter->first;
        map<int, pair<int, int> >::iterator rriter;
        cout << "router " << node << endl;
        for (rriter = riter->second.begin(); rriter != riter->second.end(); rriter++)
        {
            int other_node = rriter->first;
            int link = channel_count;
            //add the outport port assined to the map
            (riter->second)[other_node].first = outport[node];
            outport[node]++;
            cout << "\t connected to router " << other_node << " using link " << link
                 << " at outport " << rriter->second.first
                 << " lat " << rriter->second.second << endl;

            _chan[link]->SetLatency(rriter->second.second);
            _chan_cred[link]->SetLatency(rriter->second.second);

            _routers[node]->AddOutputChannel(_chan[link], _chan_cred[link]);
            _routers[other_node]->AddInputChannel(_chan[link], _chan_cred[link]);
            channel_count++;
        }
    }

    buildRoutingTable();

}


void AnyNet::RegisterRoutingFunctions()
{
    gRoutingFunctionMap["min_anynet"] = &min_anynet;
}

void min_anynet(const Router *r, const Flit *f, int in_channel,
                OutputSet *outputs, bool inject)
{
    int out_port = -1;
    if (!inject)
    {
        // assert(global_routing_table[r->GetID()].count(make_pair(f->src, f->dest)) != 0);  /// destination hit
        // out_port = global_routing_table[r->GetID()][make_pair(f->src, f->dest)];
        assert(global_routing_table[0].count(make_pair(f->src, f->dest)) != 0);  /// destination hit
        out_port = global_routing_table[0][make_pair(f->src, f->dest)][f->hops];
    }


    int vcBegin = gBeginVCs[f->cl];
    int vcEnd = gEndVCs[f->cl];

    outputs->Clear();

    outputs->AddRange(out_port, vcBegin, vcEnd);
}

void AnyNet::buildRoutingTable()
{
    cout << "========================== Routing table  =====================\n";
    /// read the routing table from file
    /// use the information of file + router_list(containing port connection information, i.e.(port, latency))
    // routing_table[(src, dst)][hop] = port
    /*
    routing_table.resize(_size);  // per router routing table
    for (auto it_rp = routing_path.begin(); it_rp != routing_path.end(); it_rp++)
    {
        pair<int, int> sd = it_rp->first;
        int src = sd.first, dst = sd.second;
        for (auto it_p = it_rp->second.begin(); it_p != it_rp->second.end(); it_p++)  // traverse the single path
        {
            if (next(it_p) == it_rp->second.end())  // dst node
            {
                routing_table[*it_p][make_pair(src, dst)] = router_list[0].at(*it_p).at(dst).first;
            }
            else  // router to router
            {
                routing_table[*it_p][make_pair(src, dst)] = router_list[1].at(*it_p).at(*(next(it_p))).first;
            }
        }
    }
    */
    for (auto it_rp = routing_path.begin(); it_rp != routing_path.end(); it_rp++)
    {
        pair<int, int> sd = it_rp->first;
        int src = sd.first, dst = sd.second;
        assert(routing_table.count(sd) == 0);
        routing_table[sd] = vector<int>();
        for (auto it_p = it_rp->second.begin(); it_p != it_rp->second.end(); it_p++)  // traverse the single path
        {
            if (next(it_p) == it_rp->second.end())  // dst node
            {
                routing_table[sd].push_back(router_list[0].at(*it_p).at(dst).first);
            }
            else  // router to router
            {
                routing_table[sd].push_back(router_list[1].at(*it_p).at(*(next(it_p))).first);
            }
        }
    }
    global_routing_table = &routing_table;
    // print the routing table
    cout << "========================== Configuration  =====================\n";
    for (auto it_rp = routing_path.begin(); it_rp != routing_path.end(); it_rp++)
    {
        int src = it_rp->first.first, dst = it_rp->first.second;
        cout << "src " << src << "->" << "dst " << dst << ":";
        for (auto it_p = it_rp->second.begin(); it_p != it_rp->second.end(); it_p++)
        {
            cout << *it_p << "  ";
        }
        cout << endl;
    }

    cout << "========================== Node Table  =====================\n";
    for (auto it = routing_table.begin(); it != routing_table.end(); it++)
    {
        cout << "source " << it->first.first << "    target " << it->first.second <<": ";
        for (auto it_p = it->second.begin(); it_p !=it->second.end();it_p++)
        {
            cout << *it_p << " ";
        }
        cout << endl;
    }
}


//11/7/2012
//basically djistra's, tested on a large dragonfly anynet configuration
void AnyNet::route(int r_start)
{
    assert(0);  // this should not be used after modifying routing algo
}


void AnyNet::readFile()
{

    ifstream network_list;
    string line;
    enum ParseState
    {
        HEAD_TYPE = 0,
        HEAD_ID,
        BODY_TYPE,
        BODY_ID,
        LINK_WEIGHT,
        /// routing table
        ROUTING_SRC,  // parse the source node
        ROUTING_DST,  // parse the destination node
        ROUTING_ROUTER  // parse the intermediate routers
    };
    enum ParseType
    {
        NODE = 0,
        ROUTER,
        UNKNOWN
    };

    network_list.open(file_name.c_str());
    if (!network_list.is_open())
    {
        cout << "Anynet:can't open network file " << file_name << endl;
        exit(-1);
    }

    //loop through the entire file
    while (!network_list.eof())
    {
        getline(network_list, line);
        if (line == "")
        {
            continue;
        }

        ParseState state = HEAD_TYPE;
        //position to parse out white sspace
        int pos = 0;
        int next_pos = -1;
        string temp;
        //the first node and its type
        int head_id = -1;
        ParseType head_type = UNKNOWN;
        //stuff that head are linked to
        ParseType body_type = UNKNOWN;
        int body_id = -1;
        int link_weight = 1;

        int node_src = -1, node_dst = -1, router_mid;
        do
        {

            //skip empty spaces
            next_pos = line.find(" ", pos);
            temp = line.substr(pos, next_pos - pos);
            pos = next_pos + 1;
            if (temp == "" || temp == " ")
            {
                continue;
            }

            switch (state)
            {
                case HEAD_TYPE:
                    if (temp == "router")
                    {
                        head_type = ROUTER;
                        state = HEAD_ID;
                    }
                    else if (temp == "node")
                    {
                        head_type = NODE;
                        state = HEAD_ID;
                    }
                    else if (temp == "*")  // begin to read the routing path
                    {
                        state = ROUTING_SRC;
                    }
                    else
                    {
                        cout << "Anynet:Unknow head of line type " << temp << "\n";
                        assert(false);
                    }

                    break;
                case HEAD_ID:
                    //need better error check
                    head_id = atoi(temp.c_str());

                    //intialize router structures
                    if (router_list[NODE].count(head_id) == 0)
                    {
                        router_list[NODE][head_id] = map<int, pair<int, int> >();
                    }
                    if (router_list[ROUTER].count(head_id) == 0)
                    {
                        router_list[ROUTER][head_id] = map<int, pair<int, int> >();
                    }

                    state = BODY_TYPE;
                    break;
                case LINK_WEIGHT:
                    if (temp == "router" ||
                        temp == "node")
                    {
                        //ignore
                    }
                    else
                    {
                        link_weight = atoi(temp.c_str());
                        router_list[head_type][head_id][body_id].second = link_weight;
                        break;
                    }
                    //intentionally letting it flow through
                case BODY_TYPE:
                    if (temp == "router")
                    {
                        body_type = ROUTER;
                    }
                    else if (temp == "node")
                    {
                        body_type = NODE;
                    }
                    else
                    {
                        cout << "Anynet:Unknow body type " << temp << "\n";
                        assert(false);
                    }
                    state = BODY_ID;
                    break;
                case BODY_ID:
                    body_id = atoi(temp.c_str());
                    //intialize router structures if necessary
                    if (body_type == ROUTER)
                    {
                        if (router_list[NODE].count(body_id) == 0)
                        {
                            router_list[NODE][body_id] = map<int, pair<int, int> >();
                        }
                        if (router_list[ROUTER].count(body_id) == 0)
                        {
                            router_list[ROUTER][body_id] = map<int, pair<int, int> >();
                        }
                    }

                    if (head_type == NODE && body_type == NODE)
                    {

                        cout << "Anynet:Cannot connect node to node " << temp << "\n";
                        assert(false);

                    }
                    else if (head_type == NODE && body_type == ROUTER)
                    {

                        if (node_list.count(head_id) != 0 &&
                            node_list[head_id] != body_id)
                        {
                            cout << "Anynet:Node " << body_id << " trying to connect to multiple router "
                                 << body_id << " and " << node_list[head_id] << endl;
                            assert(false);
                        }
                        node_list[head_id] = body_id;
                        router_list[NODE][body_id][head_id] = pair<int, int>(-1, 1);

                    }
                    else if (head_type == ROUTER && body_type == NODE)
                    {
                        //insert and check node
                        if (node_list.count(body_id) != 0 &&
                            node_list[body_id] != head_id)
                        {
                            cout << "Anynet:Node " << body_id << " trying to connect to multiple router "
                                 << body_id << " and " << node_list[head_id] << endl;
                            assert(false);
                        }
                        node_list[body_id] = head_id;
                        router_list[NODE][head_id][body_id] = pair<int, int>(-1, 1);

                    }
                    else if (head_type == ROUTER && body_type == ROUTER)
                    {
                        router_list[ROUTER][head_id][body_id] = pair<int, int>(-1, 1);
                        if (router_list[ROUTER][body_id].count(head_id) == 0)
                        {
                            router_list[ROUTER][body_id][head_id] = pair<int, int>(-1, 1);
                        }
                    }
                    state = LINK_WEIGHT;
                    break;
                case ROUTING_SRC:
                    node_src = atoi(temp.c_str());
                    assert(node_src >= 0 && node_src < int(node_list.size()));
                    state = ROUTING_DST;
                    break;
                case ROUTING_DST:
                    node_dst = atoi(temp.c_str());
                    assert(node_dst >= 0 && node_dst < int(node_list.size()));
                    state = ROUTING_ROUTER;
                    break;
                case ROUTING_ROUTER:
                    router_mid = atoi(temp.c_str());
                    assert(router_mid >= 0 && router_mid < int(router_list[0].size()));
                    routing_path[make_pair(node_src, node_dst)].push_back(router_mid);
                    break;
                default:
                    cout << "Anynet:Unknow parse state\n";
                    assert(false);
                    break;
            }

        } while (pos != 0);
        if (state != LINK_WEIGHT &&
            state != BODY_TYPE && state != ROUTING_ROUTER)
        {
            cout << "Anynet:Incomplete parse of the line: " << line << endl;
        }

    }

    //map verification, make sure the information contained in both maps
    //are the same
    assert(router_list[0].size() == router_list[1].size());

    //traffic generator assumes node list is sequential and starts at 0
    vector<int> node_check;
    for (map<int, int>::iterator i = node_list.begin();
         i != node_list.end();
         i++)
    {
        node_check.push_back(i->first);
    }
    sort(node_check.begin(), node_check.end());
    for (size_t i = 0; i < node_check.size(); i++)
    {
        if ((size_t) node_check[i] != i)
        {
            cout << "Anynet:booksim trafficmanager assumes sequential node numbering starting at 0\n";
            assert(false);
        }
    }

}

