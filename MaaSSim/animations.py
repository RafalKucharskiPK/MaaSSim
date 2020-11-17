################################################################################
# Module: utils.py
# Reusable functions to animate MaaSSim results
# Rafal Kucharski @ TU Delft
################################################################################

from matplotlib.animation import FuncAnimation
import osmnx as ox
import networkx as nx
import seaborn as sns
import matplotlib.markers as mks
import matplotlib.pyplot as plt
import PIL

import pandas as pd

import numpy as np

SIZE_VEH = 0.0015
SIZE_PAX = 0.0008

def make_veh_route(sim, veh, freq = '10s', nframes = 360):
    """
     Generates a trackpoints to animate trace of single vehicle from simulation
    :param sim: MaaSSim object with simulation results computed
    :param veh: id of vehicle
    :param freq: frequency of resampling
    :param nframes: number of frames to animate (times freq)
    :return: pandas dataframe
    """
    df = sim.runs[0].rides # get MaaSSim results
    track = df[df.veh == veh]
    track.t = pd.to_timedelta(track.t, unit='S')  # change time to pandas
    track.pos = track.pos.astype(int) # integer OSM nodes
    track['duration'] = track.t.diff() # duration of each element in sequence
    track = track.set_index(track.t) # reindex
    o = track.iloc[0].pos # first origin

    routes = list()
    for _, row in track.iterrows():  # process track
        d = row.pos # destination
        if o != d: # if moved
            dur = row.duration.total_seconds() # how long it took
            nodes = nx.shortest_path(sim.inData.G, o, d, weight='length')  # nodes traversed along shortest path
            times = pd.to_timedelta(np.arange(row.t.total_seconds()-dur, row.t.total_seconds() , dur / len(nodes)),
                                    unit='S')  # interpolate times
            for point, node in enumerate(nodes):
                routes.append([times[point], int(node), row.event, row.paxes])  # add point ofr each node
        else:
            routes.append(row[['t', 'pos', 'event', 'paxes']].values)  # add point for each event without movement
        o = d  # update to next point

    route = pd.DataFrame(routes, columns=['t', 'pos', 'event', 'paxes']).set_index('t') # collect trackpoints
    route = route.resample(freq).last().ffill()  # resample them every 10s, use last event in the period and
    # fill nans with previous events
    route['x'] = route.apply(lambda i: sim.inData.nodes.loc[i.pos].x, axis=1)  # get lon
    route['y'] = route.apply(lambda i: sim.inData.nodes.loc[i.pos].y, axis=1) # and lat of osm nodes
    waits = [0] # compute waiting times (not used)
    prev = route.pos.values[0]
    for p in route.pos.values:
        if p == prev:
            waits.append(waits[-1] + 1)
        else:
            waits.append(0)
        prev = p
    route['waits'] = waits[:-1]
    route['size'] = route.waits.apply(lambda x: min(100, max(10, x)))
    route['alpha'] = route.waits.apply(lambda x: 1 if x == 0 else 1 - min(70, x) / 100)
    route['veh'] = veh
    route['pos'] = route['pos'].astype(int)

    # refine for missing data (arifacts due to resmpling)
    route['pos_diff'] = route.pos.diff().diff()
    route['event'] = route.apply(lambda x: 'IDLE' if x.pos_diff == 0 and x.event == 'ARRIVES_AT_DROPOFF' else x.event,
                                 axis=1) # make empty and idle at the end of request
    route['paxes'] = route.apply(lambda x: x.paxes if x.event != 'IDLE' else [], axis=1)  # empty when needed
    route.event = route.event.apply(lambda x: 'IDLE' if x == 'OPENS_APP' else x) # make beginning IDLE
    return route.iloc[:nframes]  # save only first frames


def make_pax_route(sim, pax, freq='10s', nframes=360):
    """
    Generates a trackpoints to animate trace of single pax from simulation
    :param sim: MaaSSim object with simulation results computed
    :param pax: id of vehicle
    :param freq: frequency of resampling
    :param nframes: number of frames to animate (times freq)
    :return: pandas dataframe
    """
    df = sim.runs[0].trips # acces results
    track = df[df.pax == pax]
    track.t = pd.to_timedelta(track.t, unit='S') # change to times
    track.pos = track.pos.astype(int) # integer OSM nodes
    track['duration'] = track.t.diff() # get durations
    track = track.set_index(track.t) # reindex
    o = track.iloc[0].pos # first origin

    routes = list()
    for _, row in track.iterrows():  # process track
        d = row.pos # update destination
        if o != d: # if moved
            dur = row.duration.total_seconds()  # how long it took
            nodes = nx.shortest_path(sim.inData.G, o, d, weight='length') # what nodes did I pass
            times = pd.to_timedelta(np.arange(row.t.total_seconds() - dur, row.t.total_seconds(), dur / len(nodes)),
                                    unit='S') # interpolate times when nodes are visited
            for point, node in enumerate(nodes):
                routes.append([times[point], int(node), row.event]) # add trackpoints
        else:
            routes.append(row[['t', 'pos', 'event']].values)  # add point without movement
        o = d # update

    route = pd.DataFrame(routes, columns=['t', 'pos', 'event']).set_index('t') # collect results to dataframe
    route = route.resample(freq).last().ffill() # resample them every 10s, use last event in the period and
    # fill nans with previous events
    route['x'] = route.apply(lambda i: sim.inData.nodes.loc[i.pos].x, axis=1) # update psitions
    route['y'] = route.apply(lambda i: sim.inData.nodes.loc[i.pos].y, axis=1)
    waits = [0]
    prev = route.pos.values[0]
    for p in route.pos.values:
        if p == prev:
            waits.append(waits[-1] + 1)
        else:
            waits.append(0)
        prev = p
    route['waits'] = waits[:-1]
    route['size'] = route.waits.apply(lambda x: min(100, max(10, x)))
    route['alpha'] = route.waits.apply(lambda x: 1 if x == 0 else 1 - min(70, x) / 100)
    route['pax'] = pax
    route['pos'] = route['pos'].astype(int)

    return route.iloc[:nframes]



def animate(sim, veh_ids = None, pax_ids = None, do_animation = True):

    vehs = list()
    paxes = list()

    veh_labels = list()

    veh_routes = dict()
    pax_routes = dict()

    fulls = dict()
    pax_alphas = dict()

    table_vals = list()

    # determine paxes and vehs
    if pax_ids == -1:
        df = sim.runs[0].rides
        df = df[df.veh.isin(veh_ids)]
        df.paxes = df.paxes.apply(lambda x: None if len(list(x)) == 0 else list(x)[0])
        pax_ids = list(pd.Series(df.paxes.unique()).dropna().astype(int).values)
    elif pax_ids == None:
        pax_ids = list(sim.runs[0].trips.pax.unique())

    if veh_ids ==  None:
        veh_ids = list(sim.runs[0].rides.veh.unique())

    # read and resize ICONS
    ICONS = dict(p_full='../../data/p.png',
                 v_empty='../../data/v_empty.png',
                 v_full='../../data/v_full.png')

    for key in ICONS.keys():
        im = PIL.Image.open(ICONS[key]).resize([50, 50])
        ICONS[key] = np.array(im).astype(np.float) / 255

    #plot OSM graph
    fig, ax = ox.plot_graph(sim.inData.G, figsize=(10, 5), node_size=0, edge_linewidth=0.3,
                            show=False, close=False,
                            edge_color='grey', bgcolor='white')

    #style matplotlib
    plt.rcParams["font.family"] = "Helvetica"
    plt.title('MaaSSim simulation', fontsize=15)


    # inital paxes
    for pax in pax_ids:
        pax_routes[pax] = make_pax_route(sim, pax=pax) # make routes
        trackpoint = pax_routes[pax].iloc[0] # get first point
        #place icon
        paxes.append(
            ax.imshow(ICONS['p_full'], alpha = 0, extent=[trackpoint.x - SIZE_PAX / 2, trackpoint.x + SIZE_PAX / 2,
                                                          trackpoint.y - SIZE_PAX / 2, trackpoint.y + SIZE_PAX / 2]))
        pax_alphas[pax] = False

    #set clock
    clock = ax.text(x = sim.inData.nodes.x.max(),
                    y = sim.inData.nodes.y.min(),
                    s= 'time {}s'.format(int(trackpoint.name.total_seconds())))

    # initial vehicles
    for veh in veh_ids:
        veh_routes[veh] = make_veh_route(sim, veh=veh) # generate tracks
        trackpoint = veh_routes[veh].iloc[0]  # take first point
        table_vals.append([veh, trackpoint.pos,trackpoint.event,'-'])  # define values for bottom table
        veh_labels.append(ax.text(x=trackpoint.x, y=trackpoint.y, s='{}'.format(veh), fontsize=7)) # add label

        vehs.append(ax.imshow(ICONS['v_empty'], extent=[trackpoint.x-SIZE_VEH/2, trackpoint.x + SIZE_VEH/2,
                                            trackpoint.y-SIZE_VEH/2, trackpoint.y + SIZE_VEH/2]))
        fulls[veh] = False

    # plot table
    the_table = plt.table(cellText=table_vals,
                          colWidths=[0.005, 0.015,0.04,0.007],
                          colLabels=['veh','pos','event','pax'],
                          loc='bottom',
                          edges='closed')
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(7)
    the_table.scale(5, 0.8*len(veh_ids)/2)


    def init(): # for animation
        return vehs, veh_labels, the_table

    def anim(i): # animate
        for j, veh in enumerate(veh_ids):
            trackpoint = veh_routes[veh].iloc[i]
            # move
            vehs[j].set_extent([trackpoint.x-SIZE_VEH/2, trackpoint.x + SIZE_VEH/2,
                                            trackpoint.y-SIZE_VEH/2, trackpoint.y + SIZE_VEH/2])
            if len(trackpoint.paxes) > 0:
                if not fulls[veh]: # change icon to full
                    vehs[j].set_data(ICONS['v_full'])
                    fulls[veh] = True
                    veh_labels[j].set_color('white')
            elif fulls[veh]:  # change icon to empty
                vehs[j].set_data(ICONS['v_empty'])
                fulls[veh] = False
                veh_labels[j].set_color('black')
            # move label
            veh_labels[j].set_position((trackpoint.x, trackpoint.y))

            #update table
            the_table.get_celld()[(j+1, 1)].get_text().set_text(trackpoint.pos)
            the_table.get_celld()[(j+1, 2)].get_text().set_text(trackpoint.event)
            the_table.get_celld()[(j+1, 3)].get_text().set_text('-' if len(trackpoint.paxes) == 0
                                                                else trackpoint.paxes[0])
        #update clock
        clock.set_text('time: {}s'.format(int(trackpoint.name.total_seconds())))

        #move paxes
        for j, pax in enumerate(pax_ids):
            try:
                trackpoint = pax_routes[pax].iloc[i]
            except:
                trackpoint = pax_routes[pax].iloc[-1]

            # move icon
            paxes[j].set_extent([trackpoint.x - SIZE_PAX / 2, trackpoint.x + SIZE_PAX / 2,
                                 trackpoint.y - SIZE_PAX / 2, trackpoint.y + SIZE_PAX / 2])
            # make visible
            if trackpoint.event != 'STARTS_DAY':
                paxes[j].set_alpha(1)
            if trackpoint.event == 'ARRIVES_AT_DROPOFF': # disappear
                paxes[j].set_alpha(0.1)

    if do_animation:
        animation = FuncAnimation(fig, anim, frames=veh_routes[veh].shape[0], init_func=init)
        # animation.save('animation.gif', dpi=300) # to save animation
        return animation
    else:
        return fig


palette = sns.color_palette("tab10")

COLORS_VEH = dict(STARTS_DAY=palette[9],
                  RECEIVES_REQUEST=palette[0],
                  ACCEPTS_REQUEST=palette[0],
                  IS_ACCEPTED_BY_TRAVELLER='black',
                  MEETS_TRAVELLER_AT_PICKUP='black',
                  ARRIVES_AT_PICKUP=palette[2],
                  DEPARTS_FROM_PICKUP=palette[2],
                  ARRIVES_AT_DROPOFF=palette[2],
                  ENDS_SHIFT=palette[8])




marker_circle = mks.MarkerStyle('o').get_path().transformed(mks.MarkerStyle('o').get_transform())
marker_triangle = mks.MarkerStyle('>').get_path().transformed(mks.MarkerStyle('>').get_transform())
marker_cross = mks.MarkerStyle('X').get_path().transformed(mks.MarkerStyle('X').get_transform())
marker_star= mks.MarkerStyle('*').get_path().transformed(mks.MarkerStyle('*').get_transform())

MARKERS_VEH = dict(STARTS_DAY=marker_circle,
                  RECEIVES_REQUEST=marker_star,
                  ACCEPTS_REQUEST=marker_cross,
                  IS_ACCEPTED_BY_TRAVELLER=marker_cross,
                  MEETS_TRAVELLER_AT_PICKUP=marker_triangle,
                  ARRIVES_AT_PICKUP=marker_triangle,
                  DEPARTS_FROM_PICKUP=marker_triangle,
                  ARRIVES_AT_DROPOFF=marker_triangle,
                  ENDS_SHIFT=marker_circle)