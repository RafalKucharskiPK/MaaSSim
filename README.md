# MaaSSim

* (c) TU Delft, Critical MaaS ERC grant
* contributors: Rafal Kucharski, Oded Cats, Arjan de Ruijter ....

[![CodeFactor](https://www.codefactor.io/repository/github/rafalkucharskipk/maassim/badge)](https://www.codefactor.io/repository/github/rafalkucharskipk/maassim)
[![Build Status](https://travis-ci.org/RafalKucharskiPK/MaaSSim.svg?branch=master)](https://travis-ci.org/RafalKucharskiPK/MaaSSim)
[![Coverage Status](https://coveralls.io/repos/github/RafalKucharskiPK/MaaSSim/badge.svg?branch=master)](https://coveralls.io/github/RafalKucharskiPK/MaaSSim?branch=master)



>  # General MaaSSim description
> Set of passengers demand to travel between their origins and destinations at desired departure/arrival times. 
>
> Passengers may choose from number of alternative modes to supply their demand, of which we are interested in MaaS modes. 
>
> Passengers estimate utilities of their alternatives from:
> **network-wide data** (apps, planners),
> **experience** (of previous trips),
> through his **social network** (friends, ads, influencers).
>
> Classical MaaS service is a **single-passenger taxi**, which can become shared (**pooled)** and bigger (**elastic transit**).
>
> All of which may be offered by many companies and available through many platforms which may **compete** for clients. 
>
> Passenger requests to use MaaS on his trip through the **platform** from which he receives an offer that he accepts or rejects.
>
> An offer is done by a driver, by company on his behalf and inline with the strategy, or by platform inline with the policy.
>
> An offer is a pick-up place and time, arrival place and time, price and service type.
>
> When passenger-service transaction is accepted, they need to match at a given place and time, with some constrains both for passenger and driver (and other passengers inside).
> 
> Passenger travels with a driver (alone or with others) and estimates his satisfation with the service, including: time (travel, wait, walk), price and other quality factors.
> 
> Passenger conducts set of daily trips and typically optimizes decisions jointly, specifically he typically selects MaaS for a chain of trips rather than for a single trip. 
>
> Passenger optimize by selecting alternatives of maximal individual utility.
>
> Drivers, companies and platforms optimize by maximizing their profit, i.e. selecting the optimal strategy on prices, services, fleet, repositioning, marketing. 
```python
params = MaaSSim.utils.get_config('default.json') # load default config
params.city = 'Wieliczka, Poland' # modify configuration
inData = MaaSSim.utils.download_G(MaaSSim.data_structures.structures, params)  # download the graph for new city
sim_1 = MaaSSim.maassim.simulate(params = params, inData = inData) # run the simulation
params.nP = 50 # change number of travellers
sim_2 = MaaSSim.maassim.simulate(params = params, inData = inData) # run the second simulation
print(sim1.res[0].pax_kpi,sim2.res[0].pax_kpi)  # compare the results
space = DotMap()
space.nP = [50,100,200] # define the search space to explore in experiments
MaaSSim.maassim.simulate_parallel(params = params, search_space = space) # run parallel experiments
res = collect_results(params.paths.dumps) # collect results from  parallel experiments

def my_function(**kwargs): # user defined function to represent agent behaviour
    veh = kwargs.get('veh', None)  # input from simulation
    sim = veh.sim  # access to simulation object
    if len(sim.runs) > 0:
        if sim.res[last_run].veh_exp.loc[veh.id].nRIDES > 3:
            return False # if I had more than 3 rides yesterday I stay
        else:
            return True # otherwise I leave
    else:
        return True # I do not leave on first day
sim = simulate(params = params, f_driver_out = f_my_driver_out) # simulate with my user defined function
```


# MaaSSim

MaaSSim is an agent-based simulator, aiming to reproduce the transport system dynamics used by two kind of agents: (i) travellers, requesting to travel from their origin to destination at a given time, and (ii) drivers supplying their travel needs by offering them rides. The intermediate agent, the platform, allows demand to be matched with supply. Both supply and demand are microscopic. For supply this pertains to explicit representation of single vehicles and their movements in time and space (detailed road network graph), while for demand this pertains to exact trip request time and destinations defined at the graph node level.
Agents are decision makers, specifically, travellers may decide which mode they use (potentially quitting using simulated MaaS modes) or reject the incoming offer. Similarly, driver may opt-out from the system (stop providing service) or reject/accept incoming requests, moreover he may strategically reposition while being idle. All of above behaviours are modelled through user-defined modules, by default deterministic, optionally probabilistic, representing agents' taste variations (heterogeneity), their previous experiences (learning) and available information (system control). Similarly, the system performance (amongst others travel times and service times) may be deterministic or probabilistic, leading to different results' interpretations.
The simulation run results in two sets of records, one being sequence of space-time locations and statuses for simulated vehicles and the other for travellers. Further synthesized into agent-level and system-wide KPIs for further analyses.



# Installation (recommended):


* Install `Anaconda`, python 3.7+: https://www.anaconda.com/distribution/#download-section
* Run `Anaconda Navigator` 
* Create your own environment `MaaSSim`(either manually or from `environment.yml` file, see tutorial here: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file)
* Install dependencies (if env was created from `.yml` file dependencies should already be installed, otherwise install it via anaconda or `pip` (full list of imported modules is in `code/_imports.ipynb`)
* Run `Jupyter Lab` from `Anaconda Navigator`
* Clone this repo with `GitHub desktop` (https://help.github.com/en/desktop/contributing-to-projects/cloning-a-repository-from-github-desktop) to the default folder (Documents/GitHub/MaaSSim_sketches)
* Open the `/MaaSSim_sketches/code/tutorial/10 Simulator.ipynb` in `Jupyter Lab` in your browser
* Run `simulator.ipynb` and browse the code 

# Environment 
---
## prerequisites: 
---
* Python 3.6+ 
* anaconda (side libraries)
* github desktop
* (optionally) pyCharm 
    
### dependencies
---
#### jupyterLab
   accesible from anaconda navigator

#### networkX
   graph package capable of efficient graph operations, i.e. path searches (https://networkx.github.io/documentation/networkx-1.10/reference/introduction.html)

#### osmNX
   allows to donwload network (road, walk, bike, ...) from OSM via into _networkX_ python format. 
   
#### pandas
   data input and output via .csv files. 
   highly flexible and light-weight replacement for SQL-like databases.
     _DataFrame_ may store tha input and output data, handle consistency in naming, fields, structures, etc.
     
#### simpy
   multi-agent real-time simulation package with highly flexible _process_. 
   It will process:
   *  _passenger_ through the day along his decisions and trips
   *  _driver_ through the day along his decisions and trips
   * interactions betwee drivers and passengers
   http://heather.cs.ucdavis.edu/~matloff/156/PLN/DESimIntro.pdf (old but thorough intro to SimPy)
   
   
# ~~Repo structure~~  <span style="color:red">outdated</span>
~~* `code` 
    * `sandbox` meaningless trial-and error free to use and add yours
     * `sblt` code on shareability, for research and further application in simulating shared rides
     * `sim` core of the simulator
     * `utils` module-wide generic utils to use in coding
     * `tutorial` notebooks to help with getting around, some results, use-cases, etc.
     * `res` some experiments with their results
     * `pytest` to be implemented as a proper pytest
     * `params.ipynb` config file (stored as a python dict and then to .json file)
     * `_imports.ipynb` reference of all the imports and modules needed in the complete repo
     * `dataStructures.ipynb` data base definitons and schemas
     * `MaaSSim.ipynb` main *exe-like* of MassSim~~
         

# Contributing

* Start with a fresh `clone` of the latest version [cloning](https://help.github.com/en/desktop/contributing-to-projects/cloning-a-repository-from-github-desktop)
* Easiest way is to make changes is to work locally and do not push changes to the repo
* Safest way is to work on your [branch](https://help.github.com/en/desktop/contributing-to-projects/creating-a-branch-for-your-work)
* After making changes (e.g. via jupyter) you may save them locally (push) or n the server (push and commit), *remember not to push on the main branch*.
* If you make changes that do not require changes in common architecture, you may store them on your branch, or [push](https://help.github.com/en/desktop/contributing-to-projects/syncing-your-branch) them to the repo
* If you want to change something in the common architecture (change existing code instead of adding a new one) please do so via a [pull request](https://help.github.com/en/desktop/contributing-to-projects/creating-a-pull-request) that has to be reviewed and accepted
* **test** your changes by running the `0.0.1.ipynb` test and veryfing the results


# Features / functionalities (updated):

## Implemented:

* _batched and event-based platform operations_: request may be either immediately assigned to the idle driver (repsectively idle driver may be immediately assigned to queued requests), or the can be *batched* i.e. matching may be performed every _n_ seconds and pooled requests may be served together (illustrated in MaaSSim_sketches/code/tests/3 Platform batched and event based.ipynb).
* _multiplatform_: each traveller may be assigned to one or more platforms (one-to-many), each driver may be assigned to only one platform (one-to-one for simplicity). Traveller simulatneously requests to all his platforms and decides to accept/reject  incoming matches, ass soon as he is matched he pops from queues of other platofrms (MaaSSim_sketches/code/tests/4 Multiplatform.ipynb).
* _shared rides_ offline matching algorithm (presented at our seminar [here](http://www.smartptlab.tudelft.nl/index.php/sptl-seminar-7th-of-january-by-rafal-kucharski-shared-rides)). Travellers are matched before simulation into (attractive) shared-rides, then we simulate those fixed matched rides. First traveller requests the vehicle to serve the shared ride, then they visit all the nodes along the schedule.tested for Nootdorp here: Documents/GitHub/MaaSSim_sketches/code/tests/5 Multistage schedule.ipynb surprisingly computation time does not explode in such approach.
* _stochasticity_ all the durations in the simulation may be variable, controlled via sim.vars.* , e.g. sim.vars.ride = 0.1 would mean 10% variability of travel time (further explained here: Documents/GitHub/MaaSSim_sketches/code/tutorial/16 Nondeterministicity.ipynb)
* _two-level rejects for supply and demand_ both driver and may reject to be part of the system at two levels. Driver may stop being driver and traveller may stop considering any simulated MaaS modes. This happens before the simulation starts. Or, within simulation driver may reject incoming request, as well as the traveller may reject incoming driver match.
* _two sided queues_: drivers waiting for travellers or clients waiting for vehicles, tested in MaaSSim_sketches/code/tests/1 Two sided queues.ipynb with 50veh/5pax and 50pax/5veh in Noodtorp
* _tabu_: in matching (driver is matched with traveller based on shortest distance between them, yet either one of them rejects it) the search continues for the second best match.
* _rejects_: driver and vehicle are matched based on distance (by default) by default they accept, though we can simulate that they reject: traveller rejects driver (due to price, and/or waiting time) or driver rejects traveller (due to distance to pickup etc.), tested with random reject functions on (MaaSSim_sketches/code/tests/2 Rejects.ipynb).
* _saving results_ results and input files to .zip with .csv's of: requests, vehicles, rides, trips and res. Called via _sim.dump(path)_
* _transit_ we can use pre-prepared GTFS files to calculate multimodal skim matrices: transit, ride, ride->transit, transit->ride. Dependent on Ding's repo and Matplab prepared files so far, to be generalized when Uber project kicks-in (examples: MaaSSim_sketches/code/tutorial/05 Transit.ipynb) to be used in mode-choice.
* _driver shifts_ each driver by default is active throughout the simulation, though he may stard and/or end later - controllable via input .csv of drivers
* _params_ one general dict with parameters, either loaded from .csv, or populated on the fly while running and modified during simulations 
* _results_ iplemented at three levels: raw, processed and aggregated. Computed for each run and described here: MaaSSim_sketches/code/tutorial/13 Results.ipynb
* _plots and charts_ currently implemented sketches to extend: map, map+origins/destinations, passenger routes, vehicle daily routes, annotated shared rides, transit paths on map, space-time diagrams of daily pax/veh routine, ...
* multithreaded runs (via _scipy.optimize.brute(nworkers = n)_) explored over the search space (including replications). Each of scenarios is stoder vis `sim.dump`) as exemplified [here](https://github.com/RafalKucharskiPK/MaaSSim_sketches/blob/master/code/tests/7%20Parallel%20runs.ipynb)
* multiple runs supported in runner.py -> Exec(). `params.nD` iterations will be run as exemplified [here](https://github.com/RafalKucharskiPK/MaaSSim_sketches/blob/master/code/tests/8%20Iterative%20Run.ipynb)

## In progress:

* _driver D2D learning_ driver earns money and experience over consective days and may decide to quit driving. Various models and implementations by Arjan de Ruijter, hEART 2020 paper.
* _driver repositioning_ algorithm to represent idle driver behaviour for sake of his benefit (revenue/profit) - Master Thesis by Daan Knobbe, kick-off Jan 2020
* ...


## Planned/requested/awaited:


* ...

