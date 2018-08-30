## @ingroup Methods-Aerodynamics-Fidelity_Zero-Lift
# compute_max_lift_coeff.py
#
# Created:  Dec 2013, A. Variyar
# Modified: Feb 2014, T. Orra
#           Jan 2016, E. Botero         

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

#SUAVE Imports
import SUAVE
from SUAVE.Core import Units
from SUAVE.Components import Wings
from SUAVE.Core  import Data
import numpy as np

from SUAVE.Methods.Aerodynamics.Fidelity_Zero.Lift.compute_slat_lift import compute_slat_lift
from SUAVE.Methods.Aerodynamics.Fidelity_Zero.Lift.compute_flap_lift import compute_flap_lift
from SUAVE.Attributes.Gases.Air import Air

# ----------------------------------------------------------------------
#  compute_max_lift_coeff
# ----------------------------------------------------------------------

## @ingroup Methods-Aerodynamics-Fidelity_Zero-Lift
def compute_max_lift_coeff_delta(vehicle,conditions,aerodynamics):
    """Computes the maximum lift coefficient associated with an aircraft high lift system

    Assumptions:
    None

    Source:
    Unknown

    Inputs:
    vehicle.max_lift_coefficient_factor [Unitless]
    vehicle.reference_area              [m^2]
    vehicle.wings. 
      areas.reference                   [m^2]
      thickness_to_chord                [Unitless]
      chords.mean_aerodynamic           [m]
      sweeps.quarter_chord              [radians]
      taper                             [Unitless]
      flaps.chord                       [m]
      flaps.angle                       [radians]
      slats.angle                       [radians]
      areas.affected                    [m^2]
      flaps.type                        [string]
    conditions.freestream.
      velocity                          [m/s]
      density                           [kg/m^3]
      dynamic_viscosity                 [N s/m^2]

    Outputs:
    Cl_max_ls (maximum CL)              [Unitless]
    Cd_ind    (induced drag)            [Unitless]

    Properties Used:
    N/A
    """    


    # initializing Cl and CDi
    Cl_max_ls = 0
    Cd_ind    = 0

    #unpack
    max_lift_coefficient_factor = vehicle.max_lift_coefficient_factor
    angle_of_attack_limit = 14. * Units.deg
    #conditions.aerodynamics = Data()
    #conditions.aerodynamics.angle_of_attack = np.ones_like(conditions.freestream.density)*angle_of_attack_limit  
    #state = Data()
    #state.conditions = conditions
    #state.conditions.aerodynamics.lift_breakdown = Data()
    #state.conditions.aerodynamics.drag_breakdown = Data()
    #state.conditions.aerodynamics.drag_breakdown.parasite = Data()
    #results = aerodynamics.evaluate(state)
    
    state = SUAVE.Analyses.Mission.Segments.Conditions.State()
    state.conditions = SUAVE.Analyses.Mission.Segments.Conditions.Aerodynamics()
    state.conditions.freestream = conditions.freestream
    
    test_num = 1
    angle_of_attacks = np.linspace(14.000001,14.0000002,test_num)[:,None] * Units.deg
    #angle_of_attacks = np.zeros_like(angle_of_attacks)    
    
    state.expand_rows(test_num)    
        
    # --------------------------------------------------------------------
    # Initialize variables needed for CL and CD calculations
    # Use a seeded random order for values
    # --------------------------------------------------------------------
    
    ## 58,000 ft
    #Mc = np.linspace(.05,2.,test_num)
    #rho = 0.1280973981*np.ones_like(Mc)
    #mu  = 1.42E-05*np.ones_like(Mc)
    #T   = (-56.46+273.15)*np.ones_like(Mc)
    #pressure = 7961.767731*np.ones_like(Mc)
    
    ### 35,000 ft
    ##Mc = np.linspace(.05,2.,test_num)
    ##rho = 0.3808558259*np.ones_like(Mc)
    ##mu  = 1.43E-05*np.ones_like(Mc)
    ##T   = (-54.19532+273.15)*np.ones_like(Mc)
    ##pressure = 23.9191751E3*np.ones_like(Mc)   
    
    ### 0 ft
    ##Mc = np.linspace(.4,.9,test_num)
    ##rho = 1.226613787*np.ones_like(Mc)
    ##mu  = 1.79E-05*np.ones_like(Mc)
    ##T   = (15.04+273.15)*np.ones_like(Mc)
    ##pressure = 101.4009309E3*np.ones_like(Mc)     
    
    ## Changed after to preserve seed for initial testing
    #Mc = Mc[:,None]
    #rho = rho[:,None]
    #mu = mu[:,None]
    #T = T[:,None]
    #pressure = pressure[:,None]
    
    #air = Air()
    #a = air.compute_speed_of_sound(T,pressure)
    
    #re = rho*a*Mc/mu

    
    #state.conditions.freestream.mach_number = Mc
    #state.conditions.freestream.density = rho
    #state.conditions.freestream.dynamic_viscosity = mu
    #state.conditions.freestream.temperature = T
    #state.conditions.freestream.pressure = pressure
    #state.conditions.freestream.reynolds_number = re
    
    state.conditions.aerodynamics.angle_of_attack = angle_of_attacks   
    
    
    # --------------------------------------------------------------------
    # Surrogate
    # --------------------------------------------------------------------    
    
            
    #call the aero model        
    results = aerodynamics.evaluate(state)    
    Cl_max_ls = results.lift.total
    Cd_ind = results.drag.induced
        

    #Cl_max_ls = Cl_max_ls * max_lift_coefficient_factor
    return Cl_max_ls, Cd_ind


# ----------------------------------------------------------------------
#   Module Tests
# ----------------------------------------------------------------------
# this will run from command line, put simple tests for your code here
if __name__ == '__main__':

    # ------------------------------------------------------------------
    #   Initialize the Vehicle
    # ------------------------------------------------------------------    
    
    vehicle = SUAVE.Vehicle()
    vehicle.tag = 'QSST'
    
    
    # ------------------------------------------------------------------
    #   Vehicle-level Properties
    # ------------------------------------------------------------------    

    # mass properties
    mach_base = 2.2
    cruise_weight_factor=0.991-0.007*mach_base-0.01*mach_base**2
    vehicle.mass_properties.max_takeoff               = 168699. * Units.lb
    vehicle.mass_properties.operating_empty           = 50000.   # Unknown, not needed
    vehicle.mass_properties.takeoff                   = 168699. * Units.lb
    vehicle.mass_properties.cargo                     = 1000.  * Units.kilogram  # Unknown, not needed 
        
    # envelope properties
    vehicle.envelope.ultimate_load = 3.75
    vehicle.envelope.limit_load    = 2.5

    # basic parameters
    vehicle.reference_area         = 241      
    vehicle.passengers             = 55
    vehicle.systems.control        = "fully powered" 
    vehicle.systems.accessories    = "long range"
    
    
    # ------------------------------------------------------------------        
    #   Main Wing
    # ------------------------------------------------------------------        
    
    wing = SUAVE.Components.Wings.Wing()
    wing.tag = 'main_wing'
    
    wing.aspect_ratio            = 1.388
    wing.sweeps.quarter_chord    = 55.9 * Units.deg
    wing.thickness_to_chord      = 0.0225
    wing.taper                   = 0.
    wing.span_efficiency         = .9
    
    wing.spans.projected         = 18.288    
    
    wing.chords.root             = 20.8
    wing.total_length            = 21.7
    wing.chords.tip              = 2.75
    wing.chords.mean_aerodynamic = 12.03
    
    wing.areas.reference         = 241.
    wing.areas.wetted            = 344.
    wing.areas.exposed           = 210.
    wing.areas.affected          = 154.
    
    wing.twists.root             = 0.0 * Units.degrees
    wing.twists.tip              = 0.0 * Units.degrees
    
    wing.origin                  = [19.1,0,-.55]
    wing.aerodynamic_center      = [35,0,0] # not needed
    
    wing.vertical                = False
    wing.symmetric               = True
    wing.high_lift               = True
    wing.vortex_lift             = True
    wing.high_mach               = True
    
    wing.dynamic_pressure_ratio  = 1.0
    
    wing.flaps.chord_dimensional = 0.
    
    wing_airfoil = SUAVE.Components.Wings.Airfoils.Airfoil()
    wing_airfoil.coordinate_file = 'NACA65-203.dat' 
    
    wing.append_airfoil(wing_airfoil)  
    
    ## set root sweep with inner section
    #segment = SUAVE.Components.Wings.Segment()
    #segment.tag                   = 'section_1'
    #segment.percent_span_location = 0.
    #segment.twist                 = 0. * Units.deg
    #segment.root_chord_percent    = 33.8/33.8
    #segment.dihedral_outboard     = 0.
    #segment.sweeps.quarter_chord  = 67. * Units.deg
    #segment.vsp_mesh              = Data()
    #segment.vsp_mesh.inner_radius    = 1./source_ratio
    #segment.vsp_mesh.outer_radius    = 1./source_ratio
    #segment.vsp_mesh.inner_length    = .044/source_ratio
    #segment.vsp_mesh.outer_length    = .044/source_ratio
    #segment.vsp_mesh.matching_TE     = False
    #segment.append_airfoil(wing_airfoil)
    #wing.Segments.append(segment)
    
    ## set mid section start point
    #segment = SUAVE.Components.Wings.Segment()
    #segment.tag                   = 'section_2'
    #segment.percent_span_location = 6.15/(25.6/2) + wing.Segments['section_1'].percent_span_location
    #segment.twist                 = 0. * Units.deg
    #segment.root_chord_percent    = 13.8/33.8
    #segment.dihedral_outboard     = 0.
    #segment.sweeps.quarter_chord  = 48. * Units.deg
    #segment.vsp_mesh              = Data()
    #segment.vsp_mesh.inner_radius    = 1./source_ratio
    #segment.vsp_mesh.outer_radius    = .88/source_ratio
    #segment.vsp_mesh.inner_length    = .044/source_ratio
    #segment.vsp_mesh.outer_length    = .044/source_ratio 
    #segment.vsp_mesh.matching_TE     = False
    #segment.append_airfoil(wing_airfoil)
    #wing.Segments.append(segment)
    
    ## set tip section start point
    #segment = SUAVE.Components.Wings.Segment() 
    #segment.tag                   = 'section_3'
    #segment.percent_span_location = 5.95/(25.6/2) + wing.Segments['section_2'].percent_span_location
    #segment.twist                 = 0. * Units.deg
    #segment.root_chord_percent    = 4.4/33.8
    #segment.dihedral_outboard     = 0.
    #segment.sweeps.quarter_chord  = 71. * Units.deg 
    #segment.vsp_mesh              = Data()
    #segment.vsp_mesh.inner_radius    = .88/source_ratio
    #segment.vsp_mesh.outer_radius    = .22/source_ratio
    #segment.vsp_mesh.inner_length    = .044/source_ratio
    #segment.vsp_mesh.outer_length    = .011/source_ratio 
    #segment.append_airfoil(wing_airfoil)
    #wing.Segments.append(segment)    
    
    # add to vehicle
    vehicle.append_component(wing)
    
    
    # ------------------------------------------------------------------
    #   Vertical Stabilizer
    # ------------------------------------------------------------------
    
    wing = SUAVE.Components.Wings.Wing()
    wing.tag = 'vertical_stabilizer'    
    
    wing.aspect_ratio            = 0.65      #
    wing.sweeps.quarter_chord    = 59.7 * Units.deg
    wing.thickness_to_chord      = 0.035
    wing.taper                   = 0.178
    wing.span_efficiency         = 0.9
    
    wing.spans.projected         = 4.      #    

    wing.chords.root             = 11.9
    wing.total_length            = 12.
    wing.chords.tip              = 2.12
    wing.chords.mean_aerodynamic = 9.35
    
    wing.areas.reference         = 24.6    #
    wing.areas.wetted            = 60.4
    wing.areas.exposed           = 24.6
    wing.areas.affected          = 24.6
    
    wing.twists.root             = 0.0 * Units.degrees
    wing.twists.tip              = 0.0 * Units.degrees  
    
    wing.origin                  = [39.2,0,1.26]
    wing.aerodynamic_center      = [50,0,0]  # not needed  
    
    wing.vertical                = True 
    wing.symmetric               = False
    wing.t_tail                  = False
    wing.high_mach               = True     
    
    wing.dynamic_pressure_ratio  = 1.0
    
    tail_airfoil = SUAVE.Components.Wings.Airfoils.Airfoil()
    tail_airfoil.coordinate_file = 'supertail_refined.dat' 
    
    wing.append_airfoil(tail_airfoil)  

    ## set root sweep with inner section
    #segment = SUAVE.Components.Wings.Segment()
    #segment.tag                   = 'section_1'
    #segment.percent_span_location = 0.0
    #segment.twist                 = 0. * Units.deg
    #segment.root_chord_percent    = 14.5/14.5
    #segment.dihedral_outboard     = 0.
    #segment.sweeps.quarter_chord  = 63. * Units.deg
    #segment.vsp_mesh              = Data()
    #segment.vsp_mesh.inner_radius    = 2.9/source_ratio
    #segment.vsp_mesh.outer_radius    = 1.5/source_ratio
    #segment.vsp_mesh.inner_length    = .044/source_ratio
    #segment.vsp_mesh.outer_length    = .044/source_ratio
    #segment.append_airfoil(tail_airfoil)
    #wing.Segments.append(segment)
    
    ## set mid section start point
    #segment = SUAVE.Components.Wings.Segment()
    #segment.tag                   = 'section_2'
    #segment.percent_span_location = 2.4/(6.0) + wing.Segments['section_1'].percent_span_location
    #segment.twist                 = 0. * Units.deg
    #segment.root_chord_percent    = 7.5/14.5
    #segment.dihedral_outboard     = 0.
    #segment.sweeps.quarter_chord  = 40. * Units.deg
    #segment.vsp_mesh              = Data()
    #segment.vsp_mesh.inner_radius    = 1.5/source_ratio
    #segment.vsp_mesh.outer_radius    = .54/source_ratio
    #segment.vsp_mesh.inner_length    = .044/source_ratio
    #segment.vsp_mesh.outer_length    = .027/source_ratio 
    #segment.append_airfoil(tail_airfoil)
    #wing.Segments.append(segment)
    
    # add to vehicle
    vehicle.append_component(wing)    


    # ------------------------------------------------------------------
    #  Fuselage
    # ------------------------------------------------------------------
    
    fuselage = SUAVE.Components.Fuselages.Fuselage()
    fuselage.tag = 'fuselage'
    
    fuselage.seats_abreast         = 4
    fuselage.seat_pitch            = 1
    
    fuselage.fineness.nose         = 4.68
    fuselage.fineness.tail         = 4.78
    
    fuselage.lengths.total         = 51.82  
    
    fuselage.width                 = 2.43
    
    fuselage.heights.maximum       = 2.67    #
    
    fuselage.heights.maximum       = 2.67    #
    fuselage.heights.at_quarter_length              = 2.67    #
    fuselage.heights.at_wing_root_quarter_chord     = 2.67    #
    fuselage.heights.at_three_quarters_length       = 2.51    #

    fuselage.areas.wetted          = 332.
    fuselage.areas.front_projected = 5.3
    
    
    fuselage.effective_diameter    = 2.55
    
    fuselage.differential_pressure = 7.5e4 * Units.pascal    # Maximum differential pressure
    
    fuselage.OpenVSP_values = Data() # VSP uses degrees directly
    
    fuselage.OpenVSP_values.nose = Data()
    fuselage.OpenVSP_values.nose.top = Data()
    fuselage.OpenVSP_values.nose.side = Data()
    fuselage.OpenVSP_values.nose.top.angle = 20.0
    fuselage.OpenVSP_values.nose.top.strength = 0.75
    fuselage.OpenVSP_values.nose.side.angle = 20.0
    fuselage.OpenVSP_values.nose.side.strength = 0.75  
    fuselage.OpenVSP_values.nose.TB_Sym = True
    fuselage.OpenVSP_values.nose.z_pos = -.01
    
    fuselage.OpenVSP_values.tail = Data()
    fuselage.OpenVSP_values.tail.top = Data()
    fuselage.OpenVSP_values.tail.side = Data()    
    fuselage.OpenVSP_values.tail.bottom = Data()
    fuselage.OpenVSP_values.tail.top.angle = 0.0
    fuselage.OpenVSP_values.tail.top.strength = 0.0
    # after this doesn't matter in current setup
    #fuselage.OpenVSP_values.tail.side.angle = -10.0
    #fuselage.OpenVSP_values.tail.side.strength = 0.75  
    #fuselage.OpenVSP_values.tail.TB_Sym = False 
    #fuselage.OpenVSP_values.tail.bottom.angle = -20.0
    #fuselage.OpenVSP_values.tail.bottom.strength = 0.75      
    
    # add to vehicle
    vehicle.append_component(fuselage)

    conditions = Data()
    conditions.freestream = Data()
    conditions.freestream.mach_number = 0.3
    conditions.freestream.velocity    = 51. #m/s
    conditions.freestream.density     = 1.1225 #kg/m?
    conditions.freestream.dynamic_viscosity   = 1.79E-05


    Cl_max_ls, Cd_ind = compute_max_lift_coeff_delta(vehicle,conditions)
    print 'CLmax : ', Cl_max_ls, 'dCDi :' , Cd_ind

