#!/usr/bin/env python3
"""
Simple SUAVE Vehicle Definition Example

This script demonstrates the basic structure of defining a vehicle in SUAVE
without requiring external dependencies like OpenVSP.

This example shows:
1. How to create a SUAVE Vehicle object
2. How to define wings, fuselage, and other components
3. How to set up basic vehicle properties
4. How to prepare for STL export (when OpenVSP is available)

Author: Generated for SUAVE exploration
Date: 2024
"""

# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import numpy as np
import SUAVE
from SUAVE.Core import Units

# ----------------------------------------------------------------------
#   Main Function
# ----------------------------------------------------------------------

def main():
    """
    Main function that demonstrates basic vehicle definition
    """
    print("Simple SUAVE Vehicle Definition Example")
    print("=" * 45)
    
    # Define the vehicle
    print("1. Creating aircraft vehicle...")
    vehicle = create_simple_aircraft()
    print("   ✓ Vehicle created successfully")
    
    # Display vehicle information
    print("\n2. Vehicle Information:")
    print("   Vehicle tag: {}".format(vehicle.tag))
    print("   Reference area: {:.2f} m²".format(vehicle.reference_area))
    print("   Max takeoff weight: {:.1f} kg".format(vehicle.mass_properties.max_takeoff))
    print("   Number of wings: {}".format(len(vehicle.wings)))
    print("   Number of fuselages: {}".format(len(vehicle.fuselages)))
    print("   Number of networks: {}".format(len(vehicle.networks)))
    
    # Display wing details
    print("\n3. Wing Details:")
    for wing_name, wing in vehicle.wings.items():
        print("   {}:".format(wing_name))
        print("     - Span: {:.2f} m".format(wing.spans.projected))
        print("     - Root chord: {:.2f} m".format(wing.chords.root))
        print("     - Tip chord: {:.2f} m".format(wing.chords.tip))
        print("     - Aspect ratio: {:.2f}".format(wing.aspect_ratio))
        print("     - Sweep: {:.1f}°".format(wing.sweeps.quarter_chord / Units.deg))
    
    # Display fuselage details
    print("\n4. Fuselage Details:")
    for fuselage_name, fuselage in vehicle.fuselages.items():
        print("   {}:".format(fuselage_name))
        print("     - Length: {:.2f} m".format(fuselage.lengths.total))
        print("     - Width: {:.2f} m".format(fuselage.width))
        print("     - Max height: {:.2f} m".format(fuselage.heights.maximum))
    
    print("\n5. STL Export Information:")
    print("   To export STL mesh, you need:")
    print("   - OpenVSP installed (vsp or openvsp Python module)")
    print("   - Use SUAVE.Input_Output.OpenVSP.write() to create .vsp3 file")
    print("   - Use SUAVE.Input_Output.OpenVSP.write_vsp_mesh() to generate .stl")
    print("   - Example code:")
    print("     from SUAVE.Input_Output.OpenVSP import write as vsp_write")
    print("     from SUAVE.Input_Output.OpenVSP import write_vsp_mesh")
    print("     vsp_write(vehicle, 'my_aircraft')")
    print("     write_vsp_mesh(vehicle, 'my_aircraft', True, 1.2, True)")
    
    print("\nExample completed successfully!")

# ----------------------------------------------------------------------
#   Simple Aircraft Creation
# ----------------------------------------------------------------------

def create_simple_aircraft():
    """
    Create a simple aircraft vehicle in SUAVE
    
    This creates a basic single-engine aircraft with:
    - Main wing
    - Horizontal stabilizer  
    - Vertical stabilizer
    - Fuselage
    - Simple engine
    """
    
    # ------------------------------------------------------------------
    #   Initialize the Vehicle
    # ------------------------------------------------------------------
    
    vehicle = SUAVE.Vehicle()
    vehicle.tag = 'Simple_Aircraft'
    
    # ------------------------------------------------------------------
    #   Vehicle-level Properties
    # ------------------------------------------------------------------
    
    # Mass properties (in kg)
    vehicle.mass_properties.max_takeoff = 1200.0
    vehicle.mass_properties.takeoff = 1200.0
    vehicle.mass_properties.operating_empty = 800.0
    vehicle.mass_properties.max_zero_fuel = 1000.0
    vehicle.mass_properties.cargo = 0.0
    vehicle.mass_properties.center_of_gravity = [[2.0, 0.0, 0.0]]
    
    # Envelope properties
    vehicle.envelope.ultimate_load = 5.7
    vehicle.envelope.limit_load = 3.8
    
    # Design parameters
    vehicle.design_mach_number = 0.2
    vehicle.design_range = 1000.0 * Units.km
    vehicle.design_cruise_alt = 3000.0 * Units.ft
    
    # Basic parameters
    vehicle.reference_area = 16.0  # m²
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
    
    # Dimensions (in meters)
    wing.spans.projected = 11.0
    wing.chords.root = 2.0
    wing.chords.tip = 1.2
    wing.chords.mean_aerodynamic = 1.6
    
    # Areas (in m²)
    wing.areas.reference = 16.0
    wing.areas.wetted = 32.0
    
    # Twist
    wing.twists.root = 2.0 * Units.degrees
    wing.twists.tip = 0.0 * Units.degrees
    
    # Position (in meters)
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
    
    # Dimensions (in meters)
    h_stab.spans.projected = 4.0
    h_stab.chords.root = 1.0
    h_stab.chords.tip = 0.7
    h_stab.chords.mean_aerodynamic = 0.85
    
    # Areas (in m²)
    h_stab.areas.reference = 4.0
    
    # Position (in meters)
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
    
    # Dimensions (in meters)
    v_stab.spans.projected = 2.5
    v_stab.chords.root = 1.5
    v_stab.chords.tip = 0.75
    v_stab.chords.mean_aerodynamic = 1.125
    
    # Areas (in m²)
    v_stab.areas.reference = 4.2
    
    # Position (in meters)
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
    
    # Basic dimensions (in meters)
    fuselage.lengths.total = 8.0
    fuselage.lengths.nose = 1.0
    fuselage.lengths.tail = 1.0
    fuselage.lengths.cabin = 6.0
    
    # Cross-sectional dimensions (in meters)
    fuselage.width = 1.2
    fuselage.heights.maximum = 1.4
    fuselage.heights.at_quarter_length = 1.2
    fuselage.heights.at_three_quarters_length = 1.0
    fuselage.effective_diameter = 1.1
    
    # Shape parameters
    fuselage.nose_curvature = 1.5
    fuselage.tail_curvature = 1.5
    fuselage.fineness.nose = 0.8
    fuselage.fineness.tail = 0.8
    
    # Position (in meters)
    fuselage.origin = [[0.0, 0.0, 0.0]]
    fuselage.aerodynamic_center = [0.0, 0.0, 0.0]
    
    # Areas (in m²)
    fuselage.areas.front_projected = 1.68
    fuselage.areas.side_projected = 11.2
    fuselage.areas.wetted = 35.0
    
    # Add to vehicle
    vehicle.append_component(fuselage)
    
    # ------------------------------------------------------------------
    #   Simple Engine (Turbofan)
    # ------------------------------------------------------------------
    
    # Create a simple turbofan network
    turbofan = SUAVE.Components.Energy.Networks.Turbofan()
    turbofan.tag = 'turbofan'
    
    # Engine properties
    turbofan.number_of_engines = 1
    turbofan.bypass_ratio = 5.0
    turbofan.engine_length = 2.0
    turbofan.nacelle_diameter = 0.8
    
    # Engine position (in meters)
    turbofan.origin = [[6.0, 0.0, 0.0]]
    
    # Add to vehicle
    vehicle.append_component(turbofan)
    
    return vehicle

# ----------------------------------------------------------------------
#   STL Export Helper Function
# ----------------------------------------------------------------------

def export_to_stl(vehicle, filename="aircraft"):
    """
    Helper function to export vehicle to STL format
    
    This function requires OpenVSP to be installed.
    """
    try:
        from SUAVE.Input_Output.OpenVSP import write as vsp_write
        from SUAVE.Input_Output.OpenVSP import write_vsp_mesh
        
        print("Exporting to OpenVSP format...")
        vsp_write(vehicle, filename, verbose=True)
        print("✓ OpenVSP file created: {}.vsp3".format(filename))
        
        print("Generating STL mesh...")
        write_vsp_mesh(vehicle, filename, True, 1.2, True)
        print("✓ STL mesh generated: {}.stl".format(filename))
        print("✓ Key file generated: {}.key".format(filename))
        
    except ImportError:
        print("⚠ OpenVSP not available. Cannot export to STL.")
        print("  Install OpenVSP and its Python API to enable STL export.")
    except Exception as e:
        print("⚠ STL export failed: {}".format(str(e)))

# ----------------------------------------------------------------------
#   Call Main
# ----------------------------------------------------------------------

if __name__ == '__main__':
    main()