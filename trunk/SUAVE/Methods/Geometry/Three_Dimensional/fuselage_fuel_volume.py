## @ingroup Methods-Geometry-Three_Dimensional
# wing_fuel_volume.py
#
# Created:  Apr 2014, T. Orra
# Modified: Sep 2016, E. Botero

# ----------------------------------------------------------------------
#  Correlation-based methods for wing fuel capacity estimation
# ----------------------------------------------------------------------
## @ingroup Methods-Geometry-Three_Dimensional

import numpy as np

def fuselage_fuel_volume(fuselage):
    """Calculates the available fuel volume in a wing.

    Assumptions:
    None

    Source:
    

    Inputs:
    

    Outputs:
    fuselage.volume          [m^3]

    Properties Used:
    N/A
    """    
    
    width         = fuselage.effective_diameter
    total_length  = fuselage.lengths.total
    tail_fineness = fuselage.fineness.tail
    
    tail_length = width*tail_fineness
    
    tail_volume = np.pi*(width/2.)**2*tail_length/3.
    fuel_volume = 0.3*tail_volume # calibrated against the concorde
    
    return fuel_volume