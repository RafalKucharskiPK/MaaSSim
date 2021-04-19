import numpy as np
import random
import pandas as pd


def set_fixed_utilities(inData, params):
    mcp = params.mode_choice
    mset = params.alt_modes

    inData.passengers['ivt_seconds'] = inData.passengers.apply(lambda x: inData.requests.loc[x.name].ttrav.seconds,
                                                         axis=1)

    def set_U_car(row):
        car_ivt = row.ivt_seconds  # assumed same as RS
        car_cost = mset.car.km_cost * car_ivt * (params.speeds.ride / 1000) + mset.car.park_cost
        U = mcp.beta_access * mset.car.access_time + \
            mcp.beta_time_moto * car_ivt + \
            mcp.beta_cost * car_cost + \
            mcp.ASC_car
        return U

    inData.passengers['U_car'] = inData.passengers.apply(set_U_car, axis=1)

    def set_U_pt(row):
        pt_ivt = row.ivt_seconds * (params.speeds.ride / params.speeds.pt)
        pt_fare = mset.pt.base_fare + mset.pt.km_fare * pt_ivt * (params.speeds.ride / 1000)
        U = mcp.beta_access * mset.pt.access_time + \
            mcp.beta_wait_pt * mset.pt.wait_time + \
            mcp.beta_time_moto * pt_ivt + mcp.beta_cost * pt_fare + \
            mcp.ASC_pt
        return U

    inData.passengers['U_pt'] = inData.passengers.apply(set_U_pt, axis=1)

    def set_U_bike(row):


        bike_tt = row.ivt_seconds * \
                  (params.speeds.ride / params.speeds.bike)
        U = mcp.beta_time_bike * bike_tt
        return U

    inData.passengers['U_bike'] = inData.passengers.apply(set_U_bike, axis=1)

    def set_rs_fare(row):
        rs_fare = max(
            params.platforms.base_fare + params.platforms.fare * row.ivt_seconds * (params.speeds.ride / 1000),
            params.platforms.min_fare)
        return rs_fare

    inData.passengers['rs_fare'] = inData.passengers.apply(set_rs_fare, axis=1)
    inData.passengers['fixed_U_rs'] = inData.passengers.apply(lambda row: mcp.beta_time_moto * row.ivt_seconds +
                                                                    mcp.beta_cost * row.rs_fare + mcp.ASC_rs, axis=1)

    inData.passengers['exp_sum_fixed'] = inData.passengers.apply(lambda x: np.exp(x.U_car) + np.exp(x.U_pt) + np.exp(x.U_bike),
                                                     axis=1)

    return inData


def collect_experience(sim):
    params = sim.params

    "updating travellers' experience and updating new expected waiting time"
    df = sim.res[0].pax_exp[['NO_REQUEST', 'WAIT', 'LOSES_PATIENCE']]

    hist = pd.concat([~sim.res[_]['pax_exp'].NO_REQUEST for _ in range(0, 0 + 1)], axis=1, ignore_index=True)

    df['days_with_exp'] = hist.sum(axis=1)
    df['experienced'] = df['days_with_exp'] > params.evol.travellers.omega
    df['previous'] = sim.inData.passengers.expected_wait

    df['penalty'] = df.apply(
        lambda row: (~(row.NO_REQUEST) & (row.LOSES_PATIENCE > 0)) * params.evol.travellers.reject_penalty, axis=1)
    df['WAIT'] = df.apply(lambda row: max(row.penalty, row.WAIT), axis=1)
    df['kappa'] = df.apply(
        lambda row: 1 / params.evol.travellers.omega if row.experienced else 1 / (row.days_with_exp + 1), axis=1)
    df['updated'] = df.apply(lambda row: (1 - row.kappa) * row.previous + row.kappa * row.WAIT, axis=1)
    sim.inData.passengers.expected_wait = df['updated']


    return sim


def update_utils(sim):

    params = sim.params
    mcp = params.mode_choice

    sim.inData.passengers['U_rs'] = sim.inData.passengers.apply(lambda row: row.fixed_U_rs + mcp.beta_wait_rs * row.expected_wait,
                                                  axis = 1)

    sim.inData.passengers['exp_sum'] = sim.inData.passengers.apply(lambda row: np.exp(row.U_rs) + row.exp_sum_fixed,
                                                     axis=1)

    sim.inData.passengers['prob_rs'] = sim.inData.passengers.apply(lambda row: np.exp(row.U_rs)/ row.exp_sum,
                                                     axis=1)



    return sim




def trav_out_d2d(**kwargs):


    traveller = kwargs.get('pax', None)
    sim = traveller.sim

    return sim.inData.passengers.loc[traveller.id].prob_rs < random.random()
