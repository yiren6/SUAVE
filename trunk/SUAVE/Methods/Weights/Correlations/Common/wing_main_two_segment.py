## @ingroup Methods-Weights-Correlations-Common 
# wing_main_two_segment.py
#
# Created:  Oct 2018, J. Smart
# Modified: Nov 2018, J. Smart

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Core import Units
import numpy as np

# ----------------------------------------------------------------------
#   Wing Main Two Segment
# ----------------------------------------------------------------------

## @ingroup Methods-Weights-Correlations-Common 
def wing_main_two_segment(S_gross_w,b,lambda_w,t_c_w,sweep_le_1,sweep_le_2,y_c,c_r,c_c,Nult,TOW,wt_zf,rho,sigma):
    """ Calculate the wing weight of the aircraft based on the fully-stressed 
    bending weight of the wing box assuming
    
    Assumptions:
        calculated total wing weight based on a bending index and actual data 
        from 15 transport aircraft 
    
    Source: 
        N/A
        
    Inputs:
        S_gross_w - area of the wing                 [meters**2]
        b - span of the wing                         [meters**2]
        lambda_w - taper ratio of the wing           [dimensionless]
        t_c_w - thickness-to-chord ratio of the wing [dimensionless]
        sweep_le_1 - leading edge sweep of segment 1 [radians]
        sweep_le_2 - leading edge sweep of segment 2 [radians]
        y_c - spanwise segmentation location         [meters]
        c_r - wing root chord                        [meters]
        c_c - second segment root chord              [meters]
        Nult - ultimate load factor of the aircraft  [dimensionless]
        TOW - maximum takeoff weight of the aircraft [kilograms]
        wt_zf - zero fuel weight of the aircraft     [kilograms]
        rho - stressed material density              [kilograms/meters**3]
        sigma - stressed material allowable stress   [Pascals]
    
    Outputs:
        weight - weight of the wing                  [kilograms]          
        
    Properties Used:
        N/A
    """ 
    
    # unpack inputs
    span  = b / Units.ft # Convert meters to ft
    root_chord = c_r / Units.ft
    crank_chord = c_c / Units.ft
    crank_span = y_c / Units.ft
    taper = lambda_w
    t1 = sweep_le_1
    t2 = np.arctan2((c_r-c_c-np.tan(t1)*y_c),y_c)
    t3 = sweep_le_2
    t4 = np.arctan2((c_c-c_r*taper-np.tan(t3)*(b/2-y_c)),(b/2-y_c))
    theta1 = np.tan(t1) + np.tan(t2)
    theta2 = np.tan(t3) + np.tan(t4)
    area  = S_gross_w / Units.ft**2 # Convert meters squared to ft squared
    mtow  = TOW / Units.lb # Convert kg to lbs
    zfw   = wt_zf / Units.lb # Convert kg to lbs
    l_tot = Nult*np.sqrt(mtow*zfw)

    gamma = 16*l_tot*rho/(sigma*t_c_w*np.pi*span)

    #Calculate weight of wing for two-segment aircraft wing
    weight = np.abs(4.22*area + gamma * (
        (span**2/8)*(-np.log(root_chord-crank_span*theta1)/theta1 + np.log(root_chord)/theta1)
        + (span**2/8)*(-np.log(crank_chord-b/2*theta2)/theta2 + np.log(crank_chord-crank_span*theta2)/theta2)
        + ((2*root_chord**2*np.log(root_chord-crank_span*theta1)+theta1*crank_span*(2*root_chord+theta1*crank_span))/(4*theta1**3))
        - (2*root_chord**2*np.log(root_chord)/(4*theta1**3))
        + ((2*crank_chord**2*np.log(crank_chord-b/2*theta2))/(4*theta2**3))
        - ((2*crank_chord**2*np.log(crank_chord-crank_span*theta2)+theta2*crank_span*(2*crank_chord+theta2*crank_span))/(4*theta2**3)))
    )
    weight = weight * Units.lb # Convert lb to kg
 
    #print('weight: {}'.format(weight))

    return weight