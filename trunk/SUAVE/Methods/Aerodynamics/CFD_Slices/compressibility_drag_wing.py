## @ingroup Methods-Aerodynamics-CFD_Slices
# compressibility_drag_wing.py
# 
# Created:  Aug 2017, T. MacDonald
# Modified: 
#        

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# SUAVE imports
from SUAVE.Core import (
    Data, Container
)
from SUAVE.Components import Wings

# package imports
import numpy as np
import scipy as sp


# ----------------------------------------------------------------------
#  The Function
# ----------------------------------------------------------------------

## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Drag
def compressibility_drag_wing(state,settings,geometry):
    """Computes compressibility drag for a wing

    Assumptions:
    No sweep

    Source:
    None

    Inputs:
    state.conditions.
      freestream.mach_number                         [Unitless]
      aerodynamics.lift_breakdown.compressible_wings [Unitless]
    geometry.thickness_to_chord                      [Unitless]
    geometry.sweeps.quarter_chord                    [radians]

    Outputs:
    total_compressibility_drag                       [Unitless]

    Properties Used:
    N/A
    """ 
    
    # unpack
    conditions             = state.conditions
    airfoil_drag_surrogate = settings.airfoil_drag_surrogate
    surrogate_scale        = settings.surrogate_scale
    division_number        = settings.sections_per_segment
    mach_cutoff            = settings.mach_cutoff # used to eliminate points outside the surrogate range
    
    wing = geometry
    if wing.tag == 'main_wing':
        wing_lifts = conditions.aerodynamics.lift_breakdown.compressible_wings # currently the total aircraft lift
    elif wing.vertical:
        wing_lifts = 0
    else:
        wing_lifts = 0.15 * conditions.aerodynamics.lift_breakdown.compressible_wings
        
    mach           = conditions.freestream.mach_number
    velocity       = conditions.freestream.velocity
    mu             = conditions.freestream.dynamic_viscosity
    drag_breakdown = conditions.aerodynamics.drag_breakdown
    
    tc     = wing.thickness_to_chord
    root_c = wing.chords.root
    tip_c  = wing.chords.tip
    span   = wing.spans.projected
    area   = wing.areas.reference
    
    non_dim_secs = np.linspace(0,1,division_number)
    dim_secs     = non_dim_secs*span/2.
    
    chord = root_c - non_dim_secs*(root_c-tip_c)
    cl = section_cl(span, area, chord, dim_secs)
    re = chord*velocity/mu
    
    sur_mach = np.repeat(mach,len(non_dim_secs))
    sur_re   = re.reshape([len(sur_mach)]) 
    sur_tc   = np.repeat(tc,len(sur_mach))
    sur_cl   = np.tile(cl,len(mach))
    
    prediction_points = np.array([sur_mach,sur_re,sur_cl,sur_tc])
    prediction_points = prediction_points.transpose()/surrogate_scale
    
    cd_c_sec = airfoil_drag_surrogate.predict(prediction_points)
    cd_c_sec = cd_c_sec.reshape([len(mach),len(non_dim_secs)])
    
    int_y = cd_c_sec*chord
    int_x = dim_secs
    
    cd_c     = 1/area*sp.integrate.trapz(int_y,int_x)
    cd_c = np.transpose(np.array([cd_c]))
    cd_c[mach<.4] = np.zeros_like(cd_c)[mach<.4]
    
    # need to add integration step
    
    total_compressibility_drag = cd_c
    
    # dump data to conditions
    wing_results = Data(
        compressibility_drag      = cd_c    ,
        thickness_to_chord        = tc      , 
    )
    drag_breakdown.compressible[wing.tag] = wing_results
    
    return total_compressibility_drag

def section_cl(span,area,chord,y_pos):
    # assume elliptical lift distribution
    cl = 4*area/(np.pi*span*chord)*np.sqrt(1.-(2.*y_pos/span)*(2.*y_pos/span))
    return cl