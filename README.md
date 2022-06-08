# MaaSSim
### agent-based two-sided mobility platform simulator


[![CodeFactor](https://www.codefactor.io/repository/github/rafalkucharskipk/maassim/badge)](https://www.codefactor.io/repository/github/rafalkucharskipk/maassim)
[![Build Status](https://travis-ci.org/RafalKucharskiPK/MaaSSim.svg?branch=0.9.5)](https://travis-ci.org/RafalKucharskiPK/MaaSSim)
[![Coverage Status](https://coveralls.io/repos/github/RafalKucharskiPK/MaaSSim/badge.svg?branch=master)](https://coveralls.io/github/RafalKucharskiPK/MaaSSim?branch=master)

<img src="data/TU.jpg" alt="drawing" width="150"/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="data/SPTL.png" alt="drawing" width="100"/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="data/CM.png" alt="drawing" width="120"/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="data/LOGO-ERC.jpg" alt="drawing" width="80"/>




MaaSSim is an agent-based simulator, reproducing the dynamics of two-sided mobility platforms (like Uber and Lyft) in the context of urban transport networks. It models the behaviour and interactions of two kind of agents: (i) travellers, requesting to travel from their origin to destination at a given time, and (ii) drivers supplying their travel needs by offering them rides. The interactions between the two agent types are mediated by the platform, matching demand and supply. Both supply and demand are microscopic. For supply this pertains to the explicit representation of single vehicles and their movements in time and space (using a detailed road network graph), while for demand this pertains to exact trip request time and destinations defined at the graph node level. Agents are decision makers, specifically, travellers may reject the incoming offer or decide to use another mode than those offered by the mobility platform altogether. Similarly, driver may opt-out of the system (stop providing service) or reject/accept incoming requests. Moreover, they may strategically re-position while being idle. 

All of above behaviours are modelled through user-defined **decision modules**, by default deterministic, optionally probabilistic, representing agents' taste variations (heterogeneity), their previous experiences (learning) and available information (system control). 
Each simulation run results in two sets of outputs, one being the sequence of recorded space-time locations and statuses for simulated vehicles and the other for travellers. Further synthesised into agent-level and system-wide KPIs for in-depth analyses.

> Research paper with details [R. Kucharski, O. Cats: Simulating two-sided mobility platforms with MaaSSim,PLOS One, 2022](https://doi.org/10.1371/journal.pone.0269682) (https://doi.org/10.1371/journal.pone.0269682)

#### Motivation

The emergence of two-sided mobility markets disrupts the transport landscape. Conventional models for transport planning and operations are focused on top-down planning of service lines, timetable and traffic control measures which are not directly suitable for capturing the double-sided dynamics of mobility on demand services. This calls for the development of models that explicitly account for both supply-side and demand-side dynamics, as well as their interaction with the intermediate matching platform. In particular, in order to capture the bottom-up emerging order resulting from two-sided mobility, it is essential to revise the modelling approach of key elements of the transport system: demand (which is now inherently microscopic), supply (which has become a decision maker) and a road network (which capacity and congestion are no longer a single pivot variable of assignment models) along with a new agent, the platform, which orchestrates supply and demand interaction and which might be subject to regulation.

![Alt Text](data/MaaSSim_animation.gif)

## MaaSSim usage and functionalities at glance

```python
sim = MaaSSim.simulators.simulate(config = 'glance.json')  # run the simulation from a given configuration
sim.runs[0].trips  # access the results
params = MaaSSim.utils.get_config('glance.json')  # load configuration
params.city = "Nootdorp, Netherlands" # modify it
inData = MaaSSim.utils.load_G(inData,params)  # load the graph for a different city
sim_1 = MaaSSim.simulators.simulate(params=params) # run the simulation
params.nP = 5 # change number of travellers
inData = MaaSSim.utils.prep_supply_and_demand(inData, params)  # regenerate supply and demand
sim_2 = MaaSSim.simulators.simulate(inData=inData,params=params) # run the second simulation
print('Total waiting time: {}s in first simulation and {}s in the second.'.format(sim_1.res[0].pax_exp['WAIT'].sum(),
      sim_2.res[0].pax_exp['WAIT'].sum()))  # compare some results
space =  dict(nP=[5,10,20], nV = [5,10]) # define search space of supply and demand levels
MaaSSim.simulators.simulate_parallel(inData=inData, params=params, search_space = space, logger_level = logging.WARNING) # run parallel experiments
res = MaaSSim.utils.collect_results(params.paths.dumps) # collect results from  parallel experiments

def my_function(**kwargs): # user defined function to represent agent decisions
    veh = kwargs.get('veh', None)  # input from simulation
    sim = veh.sim  # access to simulation object
    if len(sim.runs) > 0:
        if sim.res[last_run].veh_exp.loc[veh.id].nRIDES > 3:
            return False # if I had more than 3 rides yesterday I stay
        else:
            return True # otherwise I leave
    else:
        return True # I do not leave on first day
        
sim = MaaSSim.simulators.simulate(inData=inData,params=params, f_driver_out = my_function, logger_level = logging.INFO) # simulate with my user defined function
```


# Overview

<img src="docs/tutorials/figs/e1a.png" alt="drawing" width="300"/><img src="docs/tutorials/figs/e1b.png" alt="drawing" width="300"/>

Fig. 1 *Average waiting times for travellers until the driver arrives (a) and for driver, until they get requested (b) in Delft.
Results from 20 replications of four hour simulation with 200 travellers and 10 vehicles in Delft, Netherlands. While travellers need
to wait longer in western part of the city, the vehicles wait for requests shorter there and their waiting times longest in eastern
part, where, in turn, traveller waiting times are shorter.*


![e2](docs/tutorials/figs/e2.png)

Fig. 2. *Service performance for various demand and supply levels. Average waiting times for traveller (left) and drivers (right). We can see opposite diagonal trends: System performance for traveller improves with increasing supply on one hand and decreasing demand on another, as travellers are served with lower waiting times. Conversely, if demand increases and fleet size decreases, the system improves for drivers, who need to wait less before requested. Yielding an interesting competitive structure, specific to two-sided platforms.*

![e3](docs/tutorials/figs/e3.png)

Fig. 3. *Searching for optimal platform competition strategy, platform competes on a market with competitor operating fleet of 20 vehicles at fare of 1.0 unit/km. We explore average vehicle kilometers per driver (a) and total platform revenues (b) resulting from varying fleet size (x-axis) and fare (per-kilometer) and 10 replications.*

![e4](docs/tutorials/figs/e4.png)

Fig. 4. *Driver reinforced learning behaviour, based on previous experience and expected outcomes, they make a daily decisions to opt out, or stay in the system. Initially high supply does not allow them to reach the desired income level, so many drivers opt out, yet as the fleet size decreases, the incomes for remaining drivers increase, making it reasonable for drivers to return to the system. Depending on user-defined configuration of learning process a realistic adaptive behaviour may be reproduced.*

<img src="docs/tutorials/figs/e5a.png" alt="drawing" width="300"/><img src="docs/tutorials/figs/e5b.png" alt="drawing" width="300"/>

Fig. 5. *Trace of rides for a single simulated vehicle without (a) and with pooled ride services (b). Segments marked with green were travelled by more than one traveller, segments marked with black were travelled empty.*


## Documentation


1. [Tutorials](https://github.com/RafalKucharskiPK/MaaSSim/tree/master/docs/tutorials):
* [Quickstart](https://github.com/RafalKucharskiPK/MaaSSim/blob/master/docs/tutorials/01_Quickstart.ipynb)
* [Overview](https://github.com/RafalKucharskiPK/MaaSSim/blob/master/docs/tutorials/00_MaaSSim_at_glance.ipynb)
* [Configuration](https://github.com/RafalKucharskiPK/MaaSSim/blob/master/docs/tutorials/A_04_Config.ipynb)
* [Your own networks](https://github.com/RafalKucharskiPK/MaaSSim/blob/master/docs/tutorials/A_01%20NetworkGraphs.ipynb)
* [You own demand](https://github.com/RafalKucharskiPK/MaaSSim/blob/master/docs/tutorials/A_03%20Synthetic%20Demand.ipynb)
* [Developing own decision functions](https://github.com/RafalKucharskiPK/MaaSSim/blob/master/docs/tutorials/06_User_defined_functionalities.ipynb)
* [Interpreting results](https://github.com/RafalKucharskiPK/MaaSSim/blob/master/docs/tutorials/05_Results.ipynb)

2. [Reproducible use-cases and experiments](https://github.com/RafalKucharskiPK/MaaSSim/tree/master/docs/Experiments)



# Installation:


* pip (not recommended)
`pip install MaaSSim` (`osmnx` has to be installed first with instructions from here [https://github.com/gboeing/osmnx#installation](https://osmnx.readthedocs.io/en/stable/#installation))
then you can use it as a module, following the cookbook from here https://github.com/RafalKucharskiPK/MaaSSim/blob/master/docs/tutorials/00_MaaSSim_at_glance.ipynb

all the files neede to run basic simuations are in te package files `MaaSSim/data`

* clone this repository (recommended) - this way you can freely adjust the code, control the inputs and configure simulations. Best way to start is then [here, in the tutorials](https://github.com/RafalKucharskiPK/MaaSSim/tree/master/docs/tutorials)

for troubleshooting the installations please see the [travis.yml](https://github.com/RafalKucharskiPK/MaaSSim/blob/master/.travis.yml) file with conda calls to create virtual env.
    
### dependencies
---
* simpy (discrete-event simulation framework)
* networkx (network graphs)
* numpy (numerical computations)
* matplotlib (plots)
* pandas (data structures)
* seaborn (visualizations)
* scipy (scientific computations)
* dotmap (data structure)
* exmas (matching trips to attractive shared rides)
* osmnx (the most trouublesome one - please follow official installation instructions [here](https://osmnx.readthedocs.io/en/stable/#installation))
    
----
Rafał Kucharski, 2022, rafal.kucharski [at] uj.edu.pl
