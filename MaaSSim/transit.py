import networkx as nx
import pandas as pd
from dotmap import DotMap
import sys
import osmnx as ox
from .utils import add_route


DINGS_PATH = "/Users/rkucharski/PycharmProjects/NetSci4PT"
sys.path.append(DINGS_PATH)
from network_loading import load_graphs


def prep_transit_graph(inData, params, calc_skim=True, plot=False):
    inData.skims = DotMap()
    inData.skims.dist = inData.skim.copy()
    inData.skims.ride = inData.skims.dist.divide(params.speeds.ride).astype(int)
    inData.skims.walk = inData.skims.dist.divide(params.speeds.walk).astype(int)


    # load graphs from
    graphs = load_graphs(params.GTFS.cities.keys(), params.GTFS.space_list, params.GTFS.transfer_penalty,
                         folder_path = params.paths.dingGTFS)

    # G_L = graphs[params.GTFS.city]['L']
    G_P = graphs[params.GTFS.city]['P']
    skims = get_skims(G_P) # create stop x stop skim matrices of ['GTC', "IVT", "WT", "TRANSFER", "NONIVT"]
    # assign nodes to stop points
    skims.pos['node'] = skims.pos.apply(lambda p: ox.get_nearest_node(inData.G, (p.y, p.x)), axis=1)
    inData.transit_stops = skims.pos.copy()

    # manipulate skims to networkx adjancency matrix
    to_concat = list()
    attrs = ['GTC', "IVT", "WT", "TRANSFER", "NONIVT"]
    for field in attrs:
        skims[field].columns = skims[field].columns.astype(int)
        adj = skims[field].stack().to_frame()
        adj.columns = [field]
        to_concat.append(adj)
    adj = pd.concat(to_concat, axis=1)
    adj['s'] = adj.index.get_level_values(0)
    adj['t'] = adj.index.get_level_values(1)
    adj['cost'] = adj['IVT'] + params.GTFS.wait_penalty * adj['WT'] + adj['TRANSFER']
    adj = adj.astype(int)
    adj['source'] = adj.apply(lambda x: skims.pos.loc[x.s].node, axis=1)
    adj['target'] = adj.apply(lambda x: skims.pos.loc[x.t].node, axis=1)

    # create transit graph (nodes are from inData.G - road graph nearest node)
    TG = nx.from_pandas_edgelist(adj, 'source', 'target', edge_attr=attrs, create_using=nx.MultiDiGraph)
    G = nx.compose(inData.G, TG) # merge road graph and transit graph

    # get graph attributes back
    to_concat = list()
    for field in attrs + ['length']:
        to_concat.append(pd.Series(nx.get_edge_attributes(G, field)))
    df = pd.concat(to_concat, axis=1)
    df.columns = attrs + ['length']
    df['L'] = df.length.fillna(999999)  # infinite length for transit links
    df = df.fillna(0)  # fill walking links with empty values

    df['WALK_TIME'] = (df.length / params.speeds.walk).astype(int)
    df['TRANSIT_COST'] = df.IVT + df.WT * 2 + df.TRANSFER + df.WALK_TIME  # either transit times or walk times

    nx.set_edge_attributes(G, df.TRANSIT_COST.to_dict(), name='TRANSIT_COST')
    nx.set_edge_attributes(G, df.L.to_dict(), name='length')

    if plot:
        import seaborn as sns
        palette = sns.color_palette("muted")
        ev = [0.003 if 'IVT' in edge[-1].keys() else 0.3 for edge in G.edges(data=True)]
        colors = ['blue' if 'IVT' in edge[-1].keys() else 'grey' for edge in G.edges(data=True)]
        fig, ax = ox.plot_graph(G, fig_height=15, fig_width=15, node_size=0, edge_linewidth=ev,
                                show=False, close=False,
                                edge_color=colors)
        ax.scatter(skims.pos.x, skims.pos.y, s=3, c='blue', marker='x')
        o, d = inData.nodes.sample(1).squeeze().name, inData.nodes.sample(1).squeeze().name
        route = nx.shortest_path(G, o, d, weight='TRANSIT_COST')
        ax = add_route(G, ax, route, color=palette[2], alpha=1,  key='length')
        route = nx.shortest_path(G, o, d, weight='length')
        ax = add_route(G, ax, route, color=palette[3], alpha=1, key='length')
    if calc_skim:
        skim_generator = nx.all_pairs_dijkstra_path_length(G, weight='TRANSIT_COST')
        skim_dict = dict(skim_generator)
        inData.skims.transit = pd.DataFrame(skim_dict).fillna(params.dist_threshold).T.astype(
            int)  # and dataframe is more intuitive
    inData.GTFS.G = G

    return inData


def get_skims(G, transfer_penalty=300, delta=0.2):
    """
    ---
    modification of Ding's compute_GTCbased_metric for matrices and not averages
    ---
    Compute the average travel impedance associated with each stop in the public
    transport network. The travel impedance is based on the generalized travel
    cost (GTC) which includes initial and transfer waiting time, in-vehicle
    times and time-equivalent transfer penalty time.

    Paramters:
    -------
    G: networkx graph object
        A weighted space-of-service graph (P-space)
    transfer_penalty: int
        A constant indicating the time-equivalent transfer penalty cost.
        The unit is second in this program
    delta: fraction
        A parameter determining the minimum percentage of the number of nodes
        that should be connected to the rest of the network. If below this
        minimum, a node is not considered a usable one in the following analysis.

    Returns
    -------
    df: dataframe

    """
    # shortest path
    sp = nx.shortest_path(G, weight='total_travel_time')
    # create a dictionary for stop travel impedance values
    # The travel impedance is also decomposed
    # GTC: total generalized travel cost
    # IVT: in-vehicle travel time
    # NONIVT: the remaining part related to transfer and waiting times
    ti = {}
    fields = ['GTC', "IVT", "WT", "TRANSFER", "NONIVT"]
    for key in sp.keys():
        ti[key] = {}
        for field in fields:
            ti[key][field] = dict()
    for source in sp.keys():
        for target in sp[source].keys():
            cur_sp = sp[source][target]
            for field in fields:
                ti[source][field][target] = 0
            if not len(cur_sp) == 1:
                # if not the node itself
                for k in range(len(cur_sp) - 1):
                    i = cur_sp[k]
                    j = cur_sp[k + 1]
                    ti[source]['IVT'][target] += G[i][j]['ivt']
                    ti[source]['WT'][target] += G[i][j]['wt']
                    ti[source]['NONIVT'][target] += G[i][j]['wt']
                ti[source]['NONIVT'][target] += (len(cur_sp) - 2) * transfer_penalty
                ti[source]['TRANSFER'][target] = (len(cur_sp) - 2) * transfer_penalty
                ti[source]['GTC'][target] = ti[source]['IVT'][target] + ti[source]['NONIVT'][target]

    skims = DotMap()
    for field in fields:
        skims[field] = pd.DataFrame([ti[i][field] for i in sp.keys()], index=sp.keys())

    x_list = list(nx.get_node_attributes(G, 'x').values())
    y_list = list(nx.get_node_attributes(G, 'y').values())
    df = pd.DataFrame({'node_id': list(sp.keys()), 'x': x_list, 'y': y_list})
    df = df.set_index('node_id')

    skims['pos'] = df
    return skims

def get_multimodal(inData, params, request, plot = True):
    o = request.origin
    d = request.destination
    params.GTFS.beta = 5

    # A find transfer stop from ride to transit
    # get two distances
    leg1 = inData.skims.ride.loc[inData.transit_stops.node][o]  # from origin to all transit_stops via road network
    leg2 = inData.skims.transit.loc[d][
        inData.transit_stops.node] / params.GTFS.beta # and from all transit_stop to destination with transit

    multimodalRT = pd.concat([leg1, leg2], axis=1)  # merge them by the stop_point (two columns of travel times)
    multimodalRT['time'] = multimodalRT.sum(axis=1) +\
                           params.GTFS.transfer_penalty  # calculate sum of travel times plus transfer penalty
    transfer_pointRT = multimodalRT.time.idxmin()  # find stop of minimal total time - this is transfer point

    # B find transfer stop from transit to ride
    leg1 = inData.skims.transit.loc[inData.transit_stops.node][
        o] / params.GTFS.beta  # transit distances from origin to stop points
    leg2 = inData.skims.ride.loc[d][inData.transit_stops.node]  # ride distances from transit stops to destination

    multimodalTR = pd.concat([leg1, leg2], axis=1)  # merge to two-clumn data frame
    multimodalTR['time'] = multimodalTR.sum(axis=1) + params.GTFS.transfer_penalty  # calc total travel and penalty
    transfer_pointTR = multimodalTR.time.idxmin()  # find transfer point of minimal time


    if plot:
        import seaborn as sns
        dists = DotMap()
        G = inData.GTFS.G
        palette = sns.color_palette("bright")
        ev = [0.001 if 'IVT' in edge[-1].keys() else 0.3 for edge in G.edges(data=True)]  # assign line weights
        colors = ['blue' if 'IVT' in edge[-1].keys() else 'grey' for edge in G.edges(data=True)] # assign color

        #plot graph
        fig, ax = ox.plot_graph(G, fig_height=15, fig_width=15, node_size=0, edge_linewidth=ev,
                                show=False, close=False,
                                edge_color=colors)
        # plot stops
        ax.scatter(inData.transit_stops.x, inData.transit_stops.y, s=3, c='black', marker='x')
        # plot O and D
        ax.annotate('O', (inData.nodes.loc[o].x * 1.0002, inData.nodes.loc[o].y * 1.00001))
        ax.annotate('D', (inData.nodes.loc[d].x * 1.0002, inData.nodes.loc[d].y * 1.00001))
        # plot two transfer points
        TRp = inData.transit_stops[inData.transit_stops.node ==transfer_pointTR].squeeze()
        ax.scatter(TRp.x, TRp.y, s=30, c='brown', marker='o')
        ax.annotate('T->R', (TRp.x * 1.0002, TRp.y * 1.00001))
        TRp = inData.transit_stops[inData.transit_stops.node == transfer_pointRT].squeeze()
        ax.scatter(TRp.x, TRp.y, s=30, c='green', marker='o')
        ax.annotate('R->T', (TRp.x * 1.0002, TRp.y * 1.00001))

        route = nx.shortest_path(G, o, d, weight='TRANSIT_COST')
        dists.transit = nx.shortest_path_length(G, o, d, weight='TRANSIT_COST') / params.GTFS.beta

        ax = add_route(G, ax, route, color='black', alpha=0.7, key='length')
        route = nx.shortest_path(G, o, d, weight='length')
        dists.ride = nx.shortest_path_length(G, o, d, weight='length')/params.speeds.ride
        ax = add_route(G, ax, route, color='blue', alpha=0.7, key='length')

        route = nx.shortest_path(G, o, transfer_pointRT, weight='length')
        ax = add_route(G, ax, route, color='green', alpha=0.7, key='length')
        route = nx.shortest_path(G, transfer_pointRT, d, weight='TRANSIT_COST')
        ax = add_route(G, ax, route, color='green', alpha=0.7, key='length')

        dists.RT_R = nx.shortest_path_length(G, o, transfer_pointRT, weight='length') / params.speeds.ride
        dists.RT_T = nx.shortest_path_length(G, transfer_pointRT, d, weight='TRANSIT_COST') / params.GTFS.beta
        dists.RT = dists.RT_R + dists.RT_T

        route = nx.shortest_path(G, o, transfer_pointTR, weight='TRANSIT_COST')
        ax = add_route(G, ax, route, color='brown', alpha=0.7, key='length')
        route = nx.shortest_path(G, transfer_pointTR, d, weight='length')
        ax = add_route(G, ax, route, color='brown', alpha=0.7, key='length')

        dists.TR_T = nx.shortest_path_length(G, o, transfer_pointTR, weight='TRANSIT_COST') /params.GTFS.beta
        dists.TR_R = nx.shortest_path_length(G, transfer_pointTR, d, weight='length')/ params.speeds.ride
        dists.TR = dists.TR_R + dists.TR_T

        return pd.Series(dists).to_frame().drop(['_typ','_subtyp'])
