#!/usr/bin/env python3
"""
SUAVE Vehicle Definition and STL Export Example

This script demonstrates how to:
1. Define a complete aircraft vehicle in SUAVE
2. Export the vehicle geometry to OpenVSP format
3. Generate STL surface mesh files

Requirements:
- SUAVE installed
- OpenVSP Python API (vsp or openvsp module)
- NumPy

Author: Generated for SUAVE exploration
Date: 2024
"""

# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import numpy as np
import SUAVE
from SUAVE.Core import Units
from SUAVE.Input_Output.OpenVSP import write as vsp_write
from SUAVE.Input_Output.OpenVSP import write_vsp_mesh

# ----------------------------------------------------------------------
#   Main Function
# ----------------------------------------------------------------------

def main():
    """
    Main function that demonstrates vehicle definition and STL export
    """
    print("SUAVE Vehicle Definition and STL Export Example")
    print("=" * 50)
    
    # Define the vehicle
    print("1. Defining aircraft vehicle...")
    vehicle = define_aircraft()
    print("   ✓ Vehicle defined successfully")
    
    # Export to OpenVSP format
    print("2. Exporting to OpenVSP format...")
    tag = "example_aircraft"
    try:
        vsp_write(vehicle, tag, verbose=True)
        print("   ✓ OpenVSP file created: {}.vsp3".format(tag))
    except Exception as e:
        print("   ⚠ Warning: OpenVSP export failed - {}".format(str(e)))
        print("   This is expected if OpenVSP is not installed")
        return
    
    # Generate STL mesh
    print("3. Generating STL surface mesh...")
    try:
        # Parameters for mesh generation
        half_mesh_flag = True      # Create half mesh with symmetry plane
        growth_ratio = 1.2         # Mesh growth ratio
        growth_limiting_flag = True # Use 3D growth limiting
        
        write_vsp_mesh(vehicle, tag, half_mesh_flag, growth_ratio, growth_limiting_flag)
        print("   ✓ STL mesh generated: {}.stl".format(tag))
        print("   ✓ Key file generated: {}.key".format(tag))
    except Exception as e:
        print("   ⚠ Warning: STL mesh generation failed - {}".format(str(e)))
        print("   This requires OpenVSP to be properly installed and configured")
    
    print("\nExample completed!")
    print("Files generated:")
    print("  - {}.vsp3 (OpenVSP geometry file)".format(tag))
    print("  - {}.stl (STL surface mesh)".format(tag))
    print("  - {}.key (Surface identification file)".format(tag))

# ----------------------------------------------------------------------
#   Aircraft Definition
# ----------------------------------------------------------------------

def define_aircraft():
    """
    Define a complete aircraft vehicle in SUAVE
    
    This creates a simple single-engine aircraft similar to a Cessna 172
    with main wing, horizontal stabilizer, vertical stabilizer, and fuselage.
    """
    
    # ------------------------------------------------------------------
    #   Initialize the Vehicle
    # ------------------------------------------------------------------
    
    vehicle = SUAVE.Vehicle()
    vehicle.tag = 'Example_Aircraft'
    
    # ------------------------------------------------------------------
    #   Vehicle-level Properties
    # ------------------------------------------------------------------
    
    # Mass properties
    vehicle.mass_properties.max_takeoff = 1200.0 * Units.kg
    vehicle.mass_properties.takeoff = 1200.0 * Units.kg
    vehicle.mass_properties.operating_empty = 800.0 * Units.kg
    vehicle.mass_properties.max_zero_fuel = 1000.0 * Units.kg
    vehicle.mass_properties.cargo = 0.0 * Units.kg
    vehicle.mass_properties.center_of_gravity = [[2.0, 0.0, 0.0]]
    
    # Envelope properties
    vehicle.envelope.ultimate_load = 5.7
    vehicle.envelope.limit_load = 3.8
    
    # Design parameters
    vehicle.design_mach_number = 0.2
    vehicle.design_range = 1000.0 * Units.km
    vehicle.design_cruise_alt = 3000.0 * Units.ft
    
    # Basic parameters
    vehicle.reference_area = 16.0 * Units.meter**2
    vehicle.passengers = 4
    
    # ------------------------------------------------------------------
    #   Main Wing
    # ------------------------------------------------------------------
    
    wing = SUAVE.Components.Wings.Main_Wing()
    wing.tag = 'main_wing'
    
    # Wing geometry
    wing.aspect_ratio = 8.0
    wing.sweeps.quarter_chord = 0.0 * Units.deg
    wing.thickness_to_chord = 0.12
    wing.taper = 0.6
    wing.dihedral = 2.0 * Units.deg
    
    # Dimensions
    wing.spans.projected = 11.0 * Units.meter
    wing.chords.root = 2.0 * Units.meter
    wing.chords.tip = 1.2 * Units.meter
    wing.chords.mean_aerodynamic = 1.6 * Units.meter
    
    # Areas
    wing.areas.reference = 16.0 * Units.meter**2
    wing.areas.wetted = 32.0 * Units.meter**2
    
    # Twist
    wing.twists.root = 2.0 * Units.degrees
    wing.twists.tip = 0.0 * Units.degrees
    
    # Position
    wing.origin = [[2.0, 0.0, 0.0]]
    wing.aerodynamic_center = [0.0, 0.0, 0.0]
    
    # Properties
    wing.vertical = False
    wing.symmetric = True
    wing.high_lift = True
    wing.dynamic_pressure_ratio = 1.0
    
    # Add wing to vehicle
    vehicle.append_component(wing)
    
    # ------------------------------------------------------------------
    #   Horizontal Stabilizer
    # ------------------------------------------------------------------
    
    h_stab = SUAVE.Components.Wings.Wing()
    h_stab.tag = 'horizontal_stabilizer'
    
    # Geometry
    h_stab.aspect_ratio = 4.0
    h_stab.sweeps.quarter_chord = 0.0 * Units.deg
    h_stab.thickness_to_chord = 0.10
    h_stab.taper = 0.7
    h_stab.dihedral = 0.0 * Units.deg
    
    # Dimensions
    h_stab.spans.projected = 4.0 * Units.meter
    h_stab.chords.root = 1.0 * Units.meter
    h_stab.chords.tip = 0.7 * Units.meter
    h_stab.chords.mean_aerodynamic = 0.85 * Units.meter
    
    # Areas
    h_stab.areas.reference = 4.0 * Units.meter**2
    
    # Position
    h_stab.origin = [[8.0, 0.0, 0.0]]
    h_stab.aerodynamic_center = [0.0, 0.0, 0.0]
    
    # Properties
    h_stab.vertical = False
    h_stab.symmetric = True
    h_stab.high_lift = False
    h_stab.dynamic_pressure_ratio = 0.9
    
    # Add to vehicle
    vehicle.append_component(h_stab)
    
    # ------------------------------------------------------------------
    #   Vertical Stabilizer
    # ------------------------------------------------------------------
    
    v_stab = SUAVE.Components.Wings.Wing()
    v_stab.tag = 'vertical_stabilizer'
    
    # Geometry
    v_stab.aspect_ratio = 1.5
    v_stab.sweeps.quarter_chord = 30.0 * Units.deg
    v_stab.thickness_to_chord = 0.10
    v_stab.taper = 0.5
    
    # Dimensions
    v_stab.spans.projected = 2.5 * Units.meter
    v_stab.chords.root = 1.5 * Units.meter
    v_stab.chords.tip = 0.75 * Units.meter
    v_stab.chords.mean_aerodynamic = 1.125 * Units.meter
    
    # Areas
    v_stab.areas.reference = 4.2 * Units.meter**2
    
    # Position
    v_stab.origin = [[8.0, 0.0, 0.0]]
    v_stab.aerodynamic_center = [0.0, 0.0, 0.0]
    
    # Properties
    v_stab.vertical = True
    v_stab.symmetric = False
    v_stab.high_lift = False
    v_stab.dynamic_pressure_ratio = 0.9
    
    # Add to vehicle
    vehicle.append_component(v_stab)
    
    # ------------------------------------------------------------------
    #   Fuselage
    # ------------------------------------------------------------------
    
    fuselage = SUAVE.Components.Fuselages.Fuselage()
    fuselage.tag = 'fuselage'
    
    # Basic dimensions
    fuselage.lengths.total = 8.0 * Units.meter
    fuselage.lengths.nose = 1.0 * Units.meter
    fuselage.lengths.tail = 1.0 * Units.meter
    fuselage.lengths.cabin = 6.0 * Units.meter
    
    # Cross-sectional dimensions
    fuselage.width = 1.2 * Units.meter
    fuselage.heights.maximum = 1.4 * Units.meter
    fuselage.heights.at_quarter_length = 1.2 * Units.meter
    fuselage.heights.at_three_quarters_length = 1.0 * Units.meter
    fuselage.effective_diameter = 1.1 * Units.meter
    
    # Shape parameters
    fuselage.nose_curvature = 1.5
    fuselage.tail_curvature = 1.5
    fuselage.fineness.nose = 0.8
    fuselage.fineness.tail = 0.8
    
    # Position
    fuselage.origin = [[0.0, 0.0, 0.0]]
    fuselage.aerodynamic_center = [0.0, 0.0, 0.0]
    
    # Areas
    fuselage.areas.front_projected = 1.68 * Units.meter**2
    fuselage.areas.side_projected = 11.2 * Units.meter**2
    fuselage.areas.wetted = 35.0 * Units.meter**2
    
    # Add to vehicle
    vehicle.append_component(fuselage)
    
    # ------------------------------------------------------------------
    #   Engine (Simple Turbofan)
    # ------------------------------------------------------------------
    
    # Create a simple turbofan network
    turbofan = SUAVE.Components.Energy.Networks.Turbofan()
    turbofan.tag = 'turbofan'
    
    # Engine properties
    turbofan.number_of_engines = 1
    turbofan.bypass_ratio = 5.0
    turbofan.engine_length = 2.0 * Units.meter
    turbofan.nacelle_diameter = 0.8 * Units.meter
    
    # Engine position
    turbofan.origin = [[6.0, 0.0, 0.0]]
    
    # Add to vehicle
    vehicle.append_component(turbofan)
    
    return vehicle

# ----------------------------------------------------------------------
#   Call Main
# ----------------------------------------------------------------------

if __name__ == '__main__':
    main()