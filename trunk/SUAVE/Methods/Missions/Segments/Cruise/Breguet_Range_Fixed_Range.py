## @ingroup Methods-Missions-Segments-Cruise
# Breguet_Range_Fixed_Range.py
# 
# Created:  Nov 2017, T. MacDonald
# Modified: 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np
from SUAVE.Core import Units, Data

# ----------------------------------------------------------------------
#  Initialize Conditions
# ----------------------------------------------------------------------

## @ingroup Methods-Missions-Segments-Cruise
def initialize_conditions(segment,state):
    """Sets the specified conditions which are given for the segment type.

    Assumptions:
    None

    Source:
    N/A

    Inputs:
    segment.altitude                            [meters]
    segment.range                               [meters]
    segment.air_speed                           [meters/second]

    Outputs:
    conditions.frames.inertial.acceleration_vector [meters/second^2]
    conditions.frames.inertial.velocity_vector     [meters/second]
    conditions.frames.inertial.position_vector     [meters]
    conditions.freestream.altitude                 [meters]
    conditions.frames.inertial.time                [seconds]

    Properties Used:
    N/A
    """      
    
    # unpack
    alt        = segment.altitude
    xf         = segment.range
    air_speed  = segment.air_speed       
    conditions = state.conditions 
    
    # check for initial altitude
    if alt is None:
        if not state.initials: raise AttributeError('altitude not set')
        alt = -1.0 * state.initials.conditions.frames.inertial.position_vector[-1,2]
        segment.altitude = alt
    
    # dimensionalize time
    t_initial = conditions.frames.inertial.time[0,0]
    t_final   = xf / air_speed + t_initial
    t_nondim  = state.numerics.dimensionless.control_points
    time      = t_nondim * (t_final-t_initial) + t_initial
    
    # pack
    state.conditions.freestream.altitude[:,0]             = alt
    state.conditions.frames.inertial.position_vector[:,2] = -alt # z points down
    state.conditions.frames.inertial.velocity_vector[:,0] = air_speed
    state.conditions.frames.inertial.time[:,0]            = time[:,0]

## @ingroup Methods-Missions-Segments-Cruise
def update_weights(segment,state):
    """Determines fuel burn according to the Breguet range equation

    Assumptions:
    No large variations in L/D or SFC during the segment

    Source:
    Common Method

    Inputs:
    conditions:
        weights.total_mass                          [kg]
        freestream.gravity                          [m/s^2]
        aerodynamics.lift_coefficient               [-]
        aerodynamics.drag_coefficient               [-]
        weights.vehicle_mass_rate                   [kg/s]
        frames.body.thrust_force_vector             [N]

    Outputs:
    conditions.frames.inertial.gravity_force_vector [N]


    Properties Used:
    N/A
    """         
    
    # unpack
    conditions = state.conditions
    mi         = conditions.weights.total_mass[0,0]
    g          = np.mean(conditions.freestream.gravity)
    Wi         = mi*g
    V          = np.mean(conditions.freestream.velocity)
    L          = np.mean(conditions.aerodynamics.lift_coefficient)
    D          = np.mean(conditions.aerodynamics.drag_coefficient)
    mdot       = np.mean(conditions.weights.vehicle_mass_rate[0,0])
    thrust     = np.mean(conditions.frames.body.thrust_force_vector[0,0])
    sfc        = mdot / thrust	    

    # final weight
    R          = segment.range
    mf         = mi/np.exp(R*g/V/(L/D)*sfc)
    Wf         = mf*g

    # pack
    conditions.weights.total_mass[:,0]                   = np.array([mi,mf])
    conditions.frames.inertial.gravity_force_vector[:,2] = np.array([Wi,Wf])

    return

def expand_state(segment,state):
    
    """Makes all vectors in the state the same size.

    Assumptions:
    N/A

    Source:
    N/A

    Inputs:
    state.numerics.number_control_points  [Unitless]

    Outputs:
    N/A

    Properties Used:
    N/A
    """       
    
    state.expand_rows(state.numerics.number_control_points)
    
    return