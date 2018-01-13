## @ingroup Components-Energy-Networks
# Propulsor_Surrogate.py
#
# Created:  Mar 2017, E. Botero
# Modified:

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave imports
import SUAVE

# package imports
import numpy as np
from SUAVE.Components.Propulsors.Propulsor import Propulsor

from SUAVE.Core import Data
import sklearn
from sklearn import gaussian_process
from sklearn import neighbors
from sklearn import svm

# ----------------------------------------------------------------------
#  Network
# ----------------------------------------------------------------------

## @ingroup Components-Energy-Networks
class Propulsor_TSFC(Propulsor):
    """ This is a way for you to load engine data from a source.
        A .csv file is read in, a surrogate made, that surrogate is used during the mission analysis.
        
        You need to use build surrogate first when setting up the vehicle to make this work.
    
        Assumptions:
        The input format for this should be Altitude, Mach, Throttle, Thrust, SFC
        
        Source:
        None
    """        
    def __defaults__(self): 
        """ This sets the default values for the network to function.
    
            Assumptions:
            None
    
            Source:
            N/A
    
            Inputs:
            None
    
            Outputs:
            None
    
            Properties Used:
            N/A
        """          
        self.tag                              = 'TSFC Propulsor'
        self.thrust_specific_fuel_consumption = 1e-5
        self.thrust_angle                     = 0.0
        self.sized_thrust                     = 1.0
    
    # manage process with a driver function
    def evaluate_thrust(self,state):
        """ Calculate thrust given the current state of the vehicle
    
            Assumptions:
            None
    
            Source:
            N/A
    
            Inputs:
            state [state()]
    
            Outputs:
            results.thrust_force_vector [newtons]
            results.vehicle_mass_rate   [kg/s]
    
            Properties Used:
            Defaulted values
        """             
        
        # Unpack the conditions
        conditions = state.conditions
        throttle   = conditions.propulsion.throttle
        
        F    = self.sized_thrust*throttle
        mdot = F*self.thrust_specific_fuel_consumption
        
        # Save the output
        results = Data()
        results.thrust_force_vector = F * [np.cos(self.thrust_angle),0,-np.sin(self.thrust_angle)]     
        results.vehicle_mass_rate   = mdot
    
        return results          
    
    def size(self,design_thrust):
        """ Build a surrogate. Multiple options for models are available including:
            -Gaussian Processes

        """          
        
        self.sized_thrust = design_thrust