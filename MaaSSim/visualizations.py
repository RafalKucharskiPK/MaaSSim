################################################################################
# Module: utils.py
# Reusable functions to visualize and plot MaaSSim results
# Rafal Kucharski @ TU Delft
################################################################################

import numpy as np
import matplotlib.pyplot as plt

import osmnx as ox
import networkx as nx

from matplotlib.collections import LineCollection


def add_route(G, ax, route, color='grey', lw=2, alpha=0.5):
    # plots route on the graph alrready plotted on ax
    edge_nodes = list(zip(route[:-1], route[1:]))
    lines = []
    for u, v in edge_nodes:
        # if there are parallel edges, select the shortest in length
        data = min(G.get_edge_data(u, v).values(), key=lambda x: x['length'])
        # if it has a geometry attribute (ie, a list of line segments)
        if 'geometry' in data:
            # add them to the list of lines to plot
            xs, ys = data['geometry'].xy
            lines.append(list(zip(xs, ys)))
        else:
            # if it doesn't have a geometry attribute, the edge is a straight
            # line from node to node
            x1 = G.nodes[u]['x']
            y1 = G.nodes[u]['y']
            x2 = G.nodes[v]['x']
            y2 = G.nodes[v]['y']
            line = [(x1, y1), (x2, y2)]
            lines.append(line)
    lc = LineCollection(lines, colors=color, linewidths=lw, alpha=alpha, zorder=3)
    ax.add_collection(lc)


def plot_map_rides(G, ts, light=True, m_size=30, lw=3):

    fig, ax = ox.plot_graph(G, figsize=(15, 15), node_size=0, edge_linewidth=0.3,
                            show=False, close=False,
                            edge_color='grey')

    colors = {1: 'orange', 2: 'teal', 3: 'maroon', 4: 'black', 5: 'green'}
    for t in ts:
        deg = t.req_id.nunique() - 1
        for i in t.req_id.dropna().unique():
            r = t[t.req_id == i]
            o = r[r.od == 'o'].iloc[0].node
            d = r[r.od == 'd'].iloc[0].node
            ax.scatter(G.nodes[o]['x'], G.nodes[o]['y'], s=m_size, c='black', marker='x')
            ax.scatter(G.nodes[d]['x'], G.nodes[d]['y'], s=m_size, c='black', marker='v')

            if not light:
                ax.annotate('o' + str(i), (G.nodes[o]['x'] * 1.0002, G.nodes[o]['y'] * 1.00001))
                ax.annotate('d' + str(i), (G.nodes[d]['x'] * 1.0002, G.nodes[d]['y'] * 1.00001))
                route = nx.shortest_path(G, o, d, weight='length')

                add_route(G, ax, route, color='black', lw=int(lw / 2), alpha=0.5)

        routes = list()  # ride segments
        o = t.node.dropna().values[0]

        for d in t.node.dropna().values[1:]:
            routes.append(nx.shortest_path(G, o, d, weight='length'))
            o = d
        for route in routes:
            add_route(G, ax, route, color=colors[deg], lw=lw, alpha=0.7)


def plot_demand(_inData, t0=None, vehicles=False, s=10, params=None):
    import matplotlib.pyplot as plt
    if t0 is None:
        t0 = _inData.requests.treq.mean()

    # plot osmnx graph, its center, scattered nodes of requests origins and destinations
    # plots requests temporal distribution
    fig, ax = plt.subplots(1, 3)
    ((t0 - _inData.requests.treq) / np.timedelta64(1, 'h')).plot.kde(title='Temporal distribution', ax=ax[0])
    (_inData.requests.ttrav / np.timedelta64(1, 'm')).plot(kind='box', title='Trips travel times [min]', ax=ax[1])
    _inData.requests.dist.plot(kind='box', title='Trips distance [m]', ax=ax[2])
    # (inData.requests.ttrav / np.timedelta64(1, 'm')).describe().to_frame().T
    plt.show()
    fig, ax = ox.plot_graph(_inData.G, figsize=(15, 15), node_size=0, edge_linewidth=0.5,
                            show=False, close=False,
                            edge_color='grey', bgcolor='white')
    for _, r in _inData.requests.iterrows():
        ax.scatter(_inData.G.nodes[r.origin]['x'], _inData.G.nodes[r.origin]['y'], c='green', s=s, marker='D')
        ax.scatter(_inData.G.nodes[r.destination]['x'], _inData.G.nodes[r.destination]['y'], c='orange', s=s)
    if vehicles:
        for _, r in _inData.vehicles.iterrows():
            ax.scatter(_inData.G.nodes[r.pos]['x'], _inData.G.nodes[r.pos]['y'], c='blue', s=s, marker='x')
    ax.scatter(_inData.G.nodes[_inData.stats['center']]['x'], _inData.G.nodes[_inData.stats['center']]['y'], c='red',
               s=10 * s, marker='+')
    plt.title(
        'Demand in {} with origins marked in green, destinations in orange and vehicles in blue'.format(params.city))
    plt.show()


def plot_veh_sim(sim, veh_id):
    t =  sim.runs[0].rides[sim.runs[0].rides.veh == veh_id]
    return plot_veh(sim.inData.G, t)

def plot_veh(G, t, m_size=30, lw=2, annotate = False):
    """
    plots a trace of vehicle rides on a graph
    :param G: osmnx graph (inData.G, or sim.inData.G)
    :param t: trips
    :param m_size: marker_size
    :param lw: linew weight
    :return: None
    """

    fig, ax = ox.plot_graph(G, figsize=(10, 10), node_size=0, edge_linewidth=0.3,
                            show=False, close=False,
                            edge_color='grey', bgcolor='white')

    t['node'] = t.pos

    degs = t.apply(lambda x: min(2,len(x.paxes)), axis=1)

    color_empty = 'lightsalmon'
    color_full = 'sienna'
    alphas = [1, 0.4, 1]
    colors = ['black', 'tab:blue', 'tab:green']

    routes = list()  # ride segments
    o = t.node.dropna().values[0]
    ax.scatter(G.nodes[o]['x'], G.nodes[o]['y'], s=m_size, c='tab:blue', marker='o')
    row = t.iloc[0]
    if annotate:
        ax.annotate("t:{}, paxes: {} {}".format(int(row.t), row.paxes, row.event),
                    (G.nodes[o]['x'] * 1.0002, G.nodes[o]['y'] * 1.00001))

    for row in t.iloc[1:].iterrows():
        d = row[1].pos
        if o != d:
            ax.scatter(G.nodes[d]['x'], G.nodes[d]['y'], s=m_size, c='tab:blue', marker='o')
            if annotate:
                ax.annotate("t:{}, paxes: {} {}".format(int(row[1].t), row[1].paxes, row[1].event),
                            (G.nodes[d]['x'] * 1.0002, G.nodes[d]['y'] * 1.00001))
        routes.append(nx.shortest_path(G, o, d, weight='length'))
        o = d
    for i, route in enumerate(routes):
        add_route(G, ax, route, color=colors[degs[i+1]], lw=lw*(1 + 3*degs[i + 1]),
                  alpha=alphas[degs[i+1]])
        # add_route(G, ax, route, color=color_empty if degs[i + 1] == 0 else color_full, lw=lw + degs[i + 1] ** 2 / 2,
        #           alpha=0.9)
    return ax

def plot_trip(sim, pax_id, run_id=None):
    from MaaSSim.traveller import travellerEvent
    G = sim.inData.G
    # space time
    if run_id is None:
        run_id = list(sim.runs.keys())[-1]
    df = sim.runs[run_id].trips
    df = df[df.pax == pax_id]
    df['status_num'] = df.apply(lambda x: travellerEvent[x.event].value, axis=1)

    fig, ax = plt.subplots()
    df.plot(x='t', y='status_num', ax=ax, drawstyle="steps-post")
    ax.yticks = plt.yticks(df.index, df.event)
    plt.show()

    # map
    routes = list()
    prev_node = df.pos.iloc[0]
    for node in df.pos[1:]:
        if prev_node != node:
            routes.append(nx.shortest_path(G, prev_node, node, weight='length'))
            routes.append(nx.shortest_path(G, prev_node, node, weight='length'))
        prev_node = node
    ox.plot_graph_routes(G, routes, node_size=0,
                         edge_color='grey', bgcolor='white')
    return ax