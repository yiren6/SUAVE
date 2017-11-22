## @ingroup Methods-Missions-Segments-Single_Point
# Set_Speed_Set_Altitude.py
# 
# Created:  Mar 2017, T. MacDonald
# Modified: Jul 2017, T. MacDonald
#           Aug 2017, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np
from SUAVE.Core import Units, Data

# ----------------------------------------------------------------------
#  Initialize Conditions
# ----------------------------------------------------------------------

## @ingroup Methods-Missions-Segments-Single_Point
def initialize_conditions(segment,state):
    """Sets the specified conditions which are given for the segment type.

    Assumptions:
    A fixed speed and altitude

    Source:
    N/A

    Inputs:
    segment.altitude                            [meters]
    segment.air_speed                           [meters/second]
    segment.x_accel                             [meters/second^2]
    segment.z_accel                             [meters/second^2]

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

## @ingroup Methods-Missions-Segments-Single_Point
def update_weights(segment,state):
    """Sets the gravity force vector during the segment

    Assumptions:
    A fixed speed and altitde

    Source:
    N/A

    Inputs:
    conditions:
        weights.total_mass                          [kilogram]
        freestream.gravity                          [meters/second^2]

    Outputs:
    conditions.frames.inertial.gravity_force_vector [newtons]


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
    mdot       = conditions.weights.vehicle_mass_rate[0,0]
    thrust     = conditions.frames.body.thrust_force_vector[0,0]
    #mdot       = mdot * Units.hr
    sfc        = mdot / thrust	    

    # final weight
    R          = segment.range
    mf         = mi/np.exp(R*g/V/(L/D)*sfc)
    Wf         = mf*g

    # pack
    conditions.weights.total_mass[:,0]                   = np.array([mi,mf])
    conditions.frames.inertial.gravity_force_vector[:,2] = np.array([Wi,Wf])

    return

def update_aerodynamics(segment,state):
    """ Gets aerodynamics conditions
    
        Assumptions:
        +X out nose
        +Y out starboard wing
        +Z down

        Inputs:
            segment.analyses.aerodynamics_model                  [Function]
            aerodynamics_model.settings.maximum_lift_coefficient [unitless]
            aerodynamics_model.geometry.reference_area           [meter^2]
            state.conditions.freestream.dynamic_pressure         [pascals]

        Outputs:
            conditions.aerodynamics.lift_coefficient [unitless]
            conditions.aerodynamics.drag_coefficient [unitless]
            conditions.frames.wind.lift_force_vector [newtons]
            conditions.frames.wind.drag_force_vector [newtons]

        Properties Used:
        N/A
    """
    
    # unpack
    conditions         = state.conditions
    aerodynamics_model = segment.analyses.aerodynamics
    q                  = state.conditions.freestream.dynamic_pressure
    Sref               = aerodynamics_model.geometry.reference_area
    g                  = conditions.freestream.gravity
   
        
    # dimensionalize
    L = state.ones_row(3) * 0.0
    D = state.ones_row(3) * 0.0
    
    LD = segment.lift_drag_ratio

    L[:,2] = -conditions.weights.total_mass[:,0]*g[:,0]
    D[:,0] = L[:,2]/LD
    
    CL = -np.transpose(np.atleast_2d(L[:,2]))/q/Sref
    CD = -np.transpose(np.atleast_2d(D[:,0]))/q/Sref
   

    # pack conditions
    conditions.aerodynamics.lift_coefficient = CL
    conditions.aerodynamics.drag_coefficient = CD
    conditions.frames.wind.lift_force_vector[:,:] = L[:,:] # z-axis
    conditions.frames.wind.drag_force_vector[:,:] = D[:,:] # x-axis
    
def update_thrust(segment,state):
    """ Evaluates the energy network to find the thrust force and mass rate

        Inputs -
            segment.analyses.energy_network    [Function]
            state                              [Data]

        Outputs -
            state.conditions:
               frames.body.thrust_force_vector [Newtons]
               weights.vehicle_mass_rate       [kg/s]


        Assumptions -


    """    
    
    ## unpack
    #energy_model = segment.analyses.energy

    ## evaluate
    #results   = energy_model.evaluate_thrust(state)
    
    thrust = -state.conditions.frames.wind.drag_force_vector[:,0]
    
    tsfc = segment.thrust_specific_fuel_consumption
    mass_flow = np.transpose(np.atleast_2d(thrust*tsfc))

    # pack conditions
    conditions = state.conditions
    conditions.frames.body.thrust_force_vector = -state.conditions.frames.wind.drag_force_vector
    conditions.weights.vehicle_mass_rate       = mass_flow

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

    n_points = 2
    
    state.expand_rows(n_points)
    
    return