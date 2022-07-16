# baseline
<hr> This project is intended to be used as a baseline for simulating controlled dynamical systems. The infrastructure
is set up to modularize various objects for controllers, estimators, and dynamical models. 

# Table of Contents
* [Team Members](#team-members)</br>
* [Getting Started](#getting_started)</br>

# <a name="team-members"></a>Team Members
* Mitchell Black <mblackjr@umich.edu>
* Bardh Hoxha <bardh.hoxha@toyota.com>
* Georgios Fainekos <georgios.fainekos@toyota.com>
* Tom Yamaguchi <tomoya.yamaguchi@toyota.com>
* Dimitra Panagou <dpanagou@umich.edu>

# <a name="getting_started"></a>Getting Started
It should be mentioned that this repository uses Python 3.8 under a virtual environment whose requirements are provided
in detail in [requirements.txt](requirements.txt). Prior to attempting to run any of the code provided, it is 
recommended that the user install the required packages via `pip install -r requirements.txt`.
</br></br>
The entry point to simulating a controlled dynamical system is [simulate.py](simulate.py). When running this file from 
the command line, the user may include a command line argument specifying which dynamical model will be simulated, 
otherwise the program will simulate whatever the default problem has been programmed to be. The simulated example will 
be saved in the problem-specific datastore folder as a pickle file (e.g. single_integrator/datastore/test.pkl for a 
single integrator problem).
</br></br>
The entry point to visualizing the results of a simulated example is [visualize.py](visualize.py). Similar to simulate, 
the visualize program will import the problem-specific parameters either from the user input or by default and will call 
on the problem's vis.py file (e.g. single_integrator/vis.py) in order to produce and save images and videos relating to
the problem.