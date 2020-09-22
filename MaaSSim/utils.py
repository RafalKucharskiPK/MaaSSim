'''
Reusable functions and methods used throughout the simulator
'''
import pandas as pd
from dotmap import DotMap
import math
import random
import logging
import numpy as np
import os

#from osmnx.utils import get_nearest_node
from osmnx.distance import get_nearest_node
import osmnx as ox
import networkx as nx
import json
from matplotlib.collections import LineCollection

from .traveller import travellerEvent
from .driver import driverEvent

# general data structure, dictionary of DataFrames with predefined columns (minimal definition)
structures = DotMap()
structures.passengers = pd.DataFrame(columns=['id', 'pos', 'event', 'platforms']).set_index('id')
structures.vehicles = pd.DataFrame(columns=['id', 'pos', 'event',
                                            'shift_start', 'shift_end', 'platform','expected_income']).set_index('id')
structures.platforms = pd.DataFrame(columns=['id', 'fare', 'name', 'batch_time']).set_index('id')
structures.requests = pd.DataFrame(columns=['pax', 'pax_id', 'origin', 'destination',
                                            'treq', 'tdep', 'ttrav', 'tarr', 'tdrop',
                                            'shareable', 'schedule_id']).set_index('pax')

structures.schedule = pd.DataFrame(columns=['id', 'node', 'time', 'req_id', 'od']).set_index('id')

def dummy_False(*args, **kwargs):
    return False


def dummy_True(*args, **kwargs):
    return True


def rand_node(df):
    # returns a random node of a graph
    return df.loc[random.choice(df.index)].name


def generic_generator(generator, n):
    # to create multiple passengers/vehicles/etc
    return pd.concat([generator(i) for i in range(1, n + 1)], axis=1, keys=range(1, n + 1)).T


def empty_series(df, id=None):
    # returns empty Series from a given DataFrame, to be used for consistency of adding new rows to DataFrames
    if id is None:
        id = len(df.index) + 1
    return pd.Series(index=df.columns, name=id)


def initialize_df(df):
    # deletes rows in DataFrame and leaves the columns and index
    # returns empty DataFrame
    if type(df) == pd.core.frame.DataFrame:
        cols = df.columns
    else:
        cols = list(df.keys())
    df = pd.DataFrame(columns=cols)
    df.index.name = 'id'
    return df


def get_config(path):
    # reads a .json file with MaaSSim configuration
    # use as: params = get_config(config.json)
    with open(path) as json_file:
        data = json.load(json_file)
        return DotMap(data)


def save_config(_params, path = None):
    if path is None:
        path = os.path.join(_params.paths.params,_params.NAME+".json")
    with open(path, "w") as write_file:
        json.dump(_params, write_file)


def set_t0(_params, now=True):
    if now:
        _params.t0 = pd.Timestamp.now().floor('1s')
    else:
        _params.t0 = pd.to_datetime(_params.t0)
    return _params


def networkstats(inData):
    """
    for a given network calculates it center of gravity (avg of node coordinates),
    gets nearest node and network radius (75th percentile of lengths from the center)
    returns a dictionary with center and radius
    """
    center_x = pd.DataFrame((inData.G.nodes(data='x')))[1].mean()
    center_y = pd.DataFrame((inData.G.nodes(data='y')))[1].mean()

    nearest = get_nearest_node(inData.G, [center_y, center_x])
    ret = DotMap({'center': nearest, 'radius': inData.skim[nearest].quantile(0.75)})
    return ret


def load_G(inData, _params=None, stats=False, set_t=True):
    # loads graph and skim from a file
    if set_t:
        _params = set_t0(_params)
    inData.G = ox.load_graphml(_params.paths.G)
    inData.nodes = pd.DataFrame.from_dict(dict(inData.G.nodes(data=True)), orient='index')
    skim = pd.read_csv(_params.paths.skim, index_col='Unnamed: 0')
    skim.columns = [int(c) for c in skim.columns]
    inData.skim = skim
    if stats:
        inData.stats = networkstats(inData)  # calculate center of network, radius and central node
    return inData


def generate_vehicles(inData_, nV):
    """
    generates single vehicle (database row with structure defined in DataStructures)
    index is consecutive number if dataframe
    position is random graph node
    status is IDLE
    """
    vehs = list()
    for i in range(nV + 1):
        vehs.append(empty_series(inData_.vehicles, id=i))

    vehs = pd.concat(vehs, axis=1, keys=range(1, nV + 1)).T
    vehs.event = driverEvent.STARTS_DAY
    vehs.platform = 0
    vehs.shift_start = 0
    vehs.shift_end = 60 * 60 * 24
    vehs.pos = vehs.pos.apply(lambda x: int(rand_node(inData_.nodes)))

    return vehs


def generate_demand(_inData, _params=None, avg_speed=False):
    # generates nP requests with a given temporal and spatial distribution of origins and destinations
    # returns _inData with dataframes requests and passengers populated.

    df = pd.DataFrame(index=np.arange(0, _params.nP), columns=_inData.passengers.columns)
    df.status = travellerEvent.STARTS_DAY
    df.pos = _inData.nodes.sample(_params.nP).index  # df.pos = df.apply(lambda x: rand_node(_inData.nodes), axis=1)
    _inData.passengers = df
    requests = pd.DataFrame(index=df.index, columns=_inData.requests.columns)
    distances = _inData.skim[_inData.stats['center']].to_frame().dropna()  # compute distances from center
    distances.columns = ['distance']
    distances = distances[distances['distance'] < _params.dist_threshold]
    distances['p_origin'] = distances['distance'].apply(lambda x:
                                                        math.exp(
                                                            _params.demand_structure.origins_dispertion * x))  # apply negative exponential distributions
    distances['p_destination'] = distances['distance'].apply(
        lambda x: math.exp(_params.demand_structure.destinations_dispertion * x))
    if _params.demand_structure.temporal_distribution == 'uniform':
        treq = np.random.uniform(-_params.simTime * 60 * 60 / 2, _params.simTime * 60 * 60 / 2,
                                 _params.nP)  # apply uniform distribution on request times
    elif _params.demand_structure.temporal_distribution == 'normal':
        treq = np.random.normal(_params.simTime * 60 * 60 / 2,
                                _params.demand_structure.temporal_dispertion * _params.simTime * 60 * 60 / 2,
                                _params.nP)  # apply normal distribution on request times
    requests.treq = [_params.t0 + pd.Timedelta(int(_), 's') for _ in treq]
    requests.origin = list(
        distances.sample(_params.nP, weights='p_origin', replace=True).index)  # sample origin nodes from a distribution
    requests.destination = list(distances.sample(_params.nP, weights='p_destination',
                                                 replace=True).index)  # sample destination nodes from a distribution

    requests['dist'] = requests.apply(lambda request: _inData.skim.loc[request.origin, request.destination], axis=1)
    while len(requests[requests.dist >= _params.dist_threshold]) > 0:
        requests.origin = requests.apply(lambda request: (distances.sample(1, weights='p_origin').index[0]
                                                          if request.dist >= _params.dist_threshold else request.origin),
                                         axis=1)
        requests.destination = requests.apply(lambda request: (distances.sample(1, weights='p_destination').index[0]
                                                               if request.dist >= _params.dist_threshold else request.destination),
                                              axis=1)
        requests.dist = requests.apply(lambda request: _inData.skim.loc[request.origin, request.destination], axis=1)

    requests['ttrav'] = requests.apply(lambda request: pd.Timedelta(request.dist, 's').floor('s'), axis=1)
    # requests.ttrav = pd.to_timedelta(requests.ttrav)
    if avg_speed:
        requests.ttrav = (pd.to_timedelta(requests.ttrav) / _params.speeds.ride).dt.floor('1s')
    requests.tarr = [request.treq + request.ttrav for _, request in requests.iterrows()]
    requests = requests.sort_values('treq')
    requests.index = df.index
    requests.pax_id = df.index
    requests.shareable = False

    _inData.requests = requests
    _inData.passengers.pos = _inData.requests.origin

    _inData.passengers.platforms = _inData.passengers.platforms.apply(lambda x: [0])

    return _inData


def prep_supply_and_demand(inData, params):
    inData = generate_demand(inData, params, avg_speed=True)
    inData.vehicles = generate_vehicles(inData, params.nV)
    inData.vehicles.platform = inData.vehicles.apply(lambda x: 0, axis=1)
    inData.passengers.platforms = inData.passengers.apply(lambda x: [0], axis=1)
    inData.requests['platform'] = inData.requests.apply(lambda row: inData.passengers.loc[row.name].platforms[0],
                                                        axis=1)

    inData.platforms = initialize_df(inData.platforms)
    inData.platforms.loc[0] = [1, 'Platform', 1]
    return inData



def plot_map_rides(G, ts, light=True, m_size=30, lw=3):
    def add_route(ax, route, color='grey', lw=2, alpha=0.5):
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

    fig, ax = ox.plot_graph(G, figsize = (15,15), node_size=0, edge_linewidth=0.3,
                            show=False, close=False,
                            edge_color='grey')

    colors = {1: 'orange', 2: 'teal', 3: 'maroon', 4: 'black', 5: 'green'}
    for t in ts:
        orig_points_lats, orig_points_lons, dest_points_lats, dest_points_lons = [], [], [], []
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

                add_route(ax, route, color='black', lw=lw / 2, alpha=0.5)

        routes = list()  # ride segments
        o = t.node.dropna().values[0]

        for d in t.node.dropna().values[1:]:
            routes.append(nx.shortest_path(G, o, d, weight='length'))
            o = d
        for route in routes:
            add_route(ax, route, color=colors[deg], lw=lw, alpha=0.7)


def plot_demand(inData, t0=None, vehicles=False, s=10, params = None):
    import matplotlib.pyplot as plt
    if t0 is None:
        t0 = inData.requests.treq.mean()

    # plot osmnx graph, its center, scattered nodes of requests origins and destinations
    # plots requests temporal distribution
    fig, ax = plt.subplots(1, 3)
    ((t0 - inData.requests.treq) / np.timedelta64(1, 'h')).plot.kde(title='Temporal distribution', ax=ax[0])
    (inData.requests.ttrav / np.timedelta64(1, 'm')).plot(kind='box', title='Trips travel times [min]', ax=ax[1])
    inData.requests.dist.plot(kind='box', title='Trips distance [m]', ax=ax[2])
    # (inData.requests.ttrav / np.timedelta64(1, 'm')).describe().to_frame().T
    plt.show()
    fig, ax = ox.plot_graph(inData.G, figsize=(15,15), node_size=0, edge_linewidth=0.5,
                            show=False, close=False,
                            edge_color='grey', bgcolor = 'white')
    for _, r in inData.requests.iterrows():
        ax.scatter(inData.G.nodes[r.origin]['x'], inData.G.nodes[r.origin]['y'], c='green', s=s, marker='D')
        ax.scatter(inData.G.nodes[r.destination]['x'], inData.G.nodes[r.destination]['y'], c='orange', s=s)
    if vehicles:
        for _, r in inData.vehicles.iterrows():
            ax.scatter(inData.G.nodes[r.pos]['x'], inData.G.nodes[r.pos]['y'], c='blue', s=s, marker='x')
    ax.scatter(inData.G.nodes[inData.stats['center']]['x'], inData.G.nodes[inData.stats['center']]['y'], c='red',
               s=10 * s, marker='+')
    plt.title(
        'Demand in {} with origins marked in green, destinations in orange and vehicles in blue'.format(params.city))
    plt.show()


def add_route(G, ax, route, color='grey', lw=2, alpha=0.5, key = 'length'):
    # plots route on the graph alrready plotted on ax
    edge_nodes = list(zip(route[:-1], route[1:]))
    lines = []
    for u, v in edge_nodes:
        # if there are parallel edges, select the shortest in length
        data = min(G.get_edge_data(u, v).values(), key=lambda x: x[key])
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
    return ax



def plot_veh(G, t, light=True, m_size=30, lw=2):
    def add_route(ax, route, color='grey', lw=2, alpha=0.5):
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

    fig, ax = ox.plot_graph(G, figsize =(15, 15), node_size=0, edge_linewidth=0.3,
                            show=False, close=False,
                            edge_color='grey', bgcolor = 'white')

    t['node'] = t.pos

    degs = t.apply(lambda x: len(x.paxes), axis=1)
    color_empty = 'lightsalmon'
    color_full = 'sienna'

    routes = list()  # ride segments
    o = t.node.dropna().values[0]
    ax.scatter(G.nodes[o]['x'], G.nodes[o]['y'], s=m_size, c='black', marker='x')
    row = t.iloc[0]
    ax.annotate("t:{}, paxes: {} {}".format(int(row.t), row.paxes, row.event),
                (G.nodes[o]['x'] * 1.0002, G.nodes[o]['y'] * 1.00001))

    for row in t.iloc[1:].iterrows():
        d = row[1].pos
        if o != d:
            ax.scatter(G.nodes[d]['x'], G.nodes[d]['y'], s=m_size, c='black', marker='x')
            ax.annotate("t:{}, paxes: {} {}".format(int(row[1].t), row[1].paxes, row[1].event),
                        (G.nodes[d]['x'] * 1.0002, G.nodes[d]['y'] * 1.00001))
        routes.append(nx.shortest_path(G, o, d, weight='length'))
        o = d
    for i, route in enumerate(routes):
        add_route(ax, route, color=color_empty if degs[i + 1] == 0 else color_full, lw=lw + degs[i + 1] ** 2 / 2,
                  alpha=0.9)


def make_config_paths(params,main = None):
    # call it whenever you change a city name, or main path
    import os
    if main is None:
        main = os.path.join(os.getcwd(),"../..")
    params.paths.main = os.path.abspath(main) # main repo folder
    params.paths.data = os.path.join(params.paths.main,'data') # data folder (not synced with repo)
    params.paths.params = os.path.join(params.paths.data,'configs')
    params.paths.postcodes = os.path.join(params.paths.data,'postcodes',"PC4_Nederland_2015.shp") # PCA4 codes shapefile
    params.paths.albatross = os.path.join(params.paths.data,'albatross') #albatross data
    params.paths.sblt = os.path.join(params.paths.data,'sblt') #sblt results
    params.paths.G = os.path.join(params.paths.data,'graphs',params.city.split(",")[0]+".graphml") #graphml of a current .city
    params.paths.skim = os.path.join(params.paths.main,'data','graphs',params.city.split(",")[0]+".csv") #csv with a skim between the nodes of the .city
    params.paths.NYC = os.path.join(params.paths.main,'data','fhv_tripdata_2018-01.csv') #csv with a skim between the nodes of the .city
    return params

def test_space():
    # to see if code works
    full_space = DotMap()
    full_space.nP = [100, 200]  # number of requests per sim time
    return full_space


def slice_space(s, replications=1, _print=False):
    # util to feed the np.optimize.brute
    def sliceme(l):
        return slice(0, len(l), 1)

    ret = list()
    sizes = list()
    size = 1
    for key in s.keys():
        ret += [sliceme(s[key])]
        sizes += [len(s[key])]
        size *= sizes[-1]
    if replications > 1:
        sizes += [replications]
        size *= sizes[-1]
        ret += [slice(0, replications, 1)]
    print('Search space to explore of dimensions {} and total size of {}'.format(sizes, size)) if _print else None
    return tuple(ret)

