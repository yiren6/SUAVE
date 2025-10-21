#!/usr/bin/env python3
"""
Example script demonstrating STL export from SUAVE vehicle using OpenVSP API

This script shows how to:
1. Create a SUAVE vehicle
2. Export it to OpenVSP format
3. Generate STL surface mesh
4. Customize mesh settings

Author: AI Assistant
Date: 2024
"""

import sys
import os
import numpy as np

# Add SUAVE to path
sys.path.append('/workspace/trunk')

import SUAVE
from SUAVE.Core import Units, Data
from SUAVE.Input_Output.OpenVSP.write_vsp_mesh import write_vsp_mesh
from SUAVE.Input_Output.OpenVSP.vsp_write import write as write_vsp

def create_simple_aircraft():
    """Create a simple SUAVE aircraft for demonstration"""
    
    # Initialize vehicle
    vehicle = SUAVE.Vehicle()
    vehicle.tag = 'Simple_Aircraft'
    
    # Basic vehicle properties
    vehicle.mass_properties.max_takeoff = 10000.0  # kg
    vehicle.reference_area = 50.0  # m^2
    
    # Create main wing
    wing = SUAVE.Components.Wings.Main_Wing()
    wing.tag = 'main_wing'
    
    # Wing geometry
    wing.aspect_ratio = 8.0
    wing.sweeps.quarter_chord = 20.0 * Units.degrees
    wing.thickness_to_chord = 0.12
    wing.taper = 0.3
    wing.spans.projected = 20.0  # m
    wing.chords.root = 3.0  # m
    wing.chords.tip = 0.9  # m
    wing.chords.mean_aerodynamic = 2.0  # m
    wing.areas.reference = 50.0  # m^2
    wing.twists.root = 2.0 * Units.degrees
    wing.twists.tip = 0.0 * Units.degrees
    wing.origin = [[5.0, 0.0, 0.0]]
    wing.vertical = False
    wing.symmetric = True
    
    # Add wing segments for better mesh control
    root_segment = SUAVE.Components.Wings.Segment()
    root_segment.tag = 'root_segment'
    root_segment.percent_span_location = 0.0
    root_segment.twist = 2.0 * Units.degrees
    root_segment.root_chord_percent = 1.0
    root_segment.dihedral_outboard = 0.0 * Units.degrees
    root_segment.sweeps.quarter_chord = 20.0 * Units.degrees
    root_segment.thickness_to_chord = 0.12
    
    tip_segment = SUAVE.Components.Wings.Segment()
    tip_segment.tag = 'tip_segment'
    tip_segment.percent_span_location = 1.0
    tip_segment.twist = 0.0 * Units.degrees
    tip_segment.root_chord_percent = 0.3
    tip_segment.dihedral_outboard = 0.0 * Units.degrees
    tip_segment.sweeps.quarter_chord = 20.0 * Units.degrees
    tip_segment.thickness_to_chord = 0.12
    
    wing.Segments.append(root_segment)
    wing.Segments.append(tip_segment)
    
    # Add custom mesh settings for better control
    wing.Segments[0].vsp_mesh = Data()
    wing.Segments[0].vsp_mesh.inner_length = 0.02  # m
    wing.Segments[0].vsp_mesh.outer_length = 0.02  # m
    wing.Segments[0].vsp_mesh.inner_radius = 0.4   # m
    wing.Segments[0].vsp_mesh.outer_radius = 0.4   # m
    wing.Segments[0].vsp_mesh.matching_TE = True
    
    wing.Segments[1].vsp_mesh = Data()
    wing.Segments[1].vsp_mesh.inner_length = 0.015  # m
    wing.Segments[1].vsp_mesh.outer_length = 0.015  # m
    wing.Segments[1].vsp_mesh.inner_radius = 0.3    # m
    wing.Segments[1].vsp_mesh.outer_radius = 0.3    # m
    wing.Segments[1].vsp_mesh.matching_TE = True
    
    vehicle.append_component(wing)
    
    # Create fuselage
    fuselage = SUAVE.Components.Fuselages.Fuselage()
    fuselage.tag = 'fuselage'
    
    # Fuselage geometry
    fuselage.width = 2.0  # m
    fuselage.lengths.total = 15.0  # m
    fuselage.heights.maximum = 2.5  # m
    fuselage.effective_diameter = 2.0  # m
    fuselage.fineness.nose = 2.0
    fuselage.fineness.tail = 2.0
    fuselage.origin = [[0.0, 0.0, 0.0]]
    
    # Add custom mesh settings for fuselage
    fuselage.vsp_mesh = Data()
    fuselage.vsp_mesh.length = 0.3  # m
    fuselage.vsp_mesh.radius = 0.4  # m
    
    vehicle.append_component(fuselage)
    
    return vehicle

def export_vehicle_to_stl(vehicle, output_tag, mesh_quality='medium'):
    """
    Export SUAVE vehicle to STL format using OpenVSP
    
    Parameters:
    -----------
    vehicle : SUAVE.Vehicle
        SUAVE vehicle object
    output_tag : str
        Output filename tag (without extension)
    mesh_quality : str
        Mesh quality: 'coarse', 'medium', 'fine'
    """
    
    print(f"Exporting vehicle '{vehicle.tag}' to STL format...")
    
    # Define mesh quality settings
    mesh_settings = {
        'coarse': {
            'growth_ratio': 1.3,
            'growth_limiting': False,
            'half_mesh': True
        },
        'medium': {
            'growth_ratio': 1.2,
            'growth_limiting': True,
            'half_mesh': True
        },
        'fine': {
            'growth_ratio': 1.1,
            'growth_limiting': True,
            'half_mesh': True
        }
    }
    
    settings = mesh_settings[mesh_quality]
    
    try:
        # Step 1: Export vehicle to OpenVSP format
        print("Step 1: Converting SUAVE vehicle to OpenVSP format...")
        write_vsp(vehicle, output_tag, verbose=True, write_file=True)
        print(f"OpenVSP file created: {output_tag}.vsp3")
        
        # Step 2: Generate STL mesh
        print("Step 2: Generating STL surface mesh...")
        write_vsp_mesh(
            geometry=vehicle,
            tag=output_tag,
            half_mesh_flag=settings['half_mesh'],
            growth_ratio=settings['growth_ratio'],
            growth_limiting_flag=settings['growth_limiting']
        )
        
        print(f"STL mesh generated: {output_tag}.stl")
        print(f"Key file generated: {output_tag}.key")
        
        return True
        
    except Exception as e:
        print(f"Error during STL export: {str(e)}")
        return False

def demonstrate_mesh_quality_comparison(vehicle):
    """Demonstrate different mesh quality settings"""
    
    print("\n" + "="*60)
    print("MESH QUALITY COMPARISON")
    print("="*60)
    
    qualities = ['coarse', 'medium', 'fine']
    
    for quality in qualities:
        print(f"\nGenerating {quality} mesh...")
        success = export_vehicle_to_stl(vehicle, f"aircraft_{quality}", quality)
        
        if success:
            print(f"✓ {quality.capitalize()} mesh completed successfully")
        else:
            print(f"✗ {quality.capitalize()} mesh failed")

def demonstrate_component_specific_settings(vehicle):
    """Demonstrate component-specific mesh settings"""
    
    print("\n" + "="*60)
    print("COMPONENT-SPECIFIC MESH SETTINGS")
    print("="*60)
    
    # Show how to access and modify mesh settings
    for wing in vehicle.wings:
        print(f"\nWing: {wing.tag}")
        for i, segment in enumerate(wing.Segments):
            if hasattr(segment, 'vsp_mesh'):
                print(f"  Segment {i}:")
                print(f"    Inner length: {segment.vsp_mesh.inner_length} m")
                print(f"    Outer length: {segment.vsp_mesh.outer_length} m")
                print(f"    Inner radius: {segment.vsp_mesh.inner_radius} m")
                print(f"    Outer radius: {segment.vsp_mesh.outer_radius} m")
    
    for fuselage in vehicle.fuselages:
        print(f"\nFuselage: {fuselage.tag}")
        if hasattr(fuselage, 'vsp_mesh'):
            print(f"  Length: {fuselage.vsp_mesh.length} m")
            print(f"  Radius: {fuselage.vsp_mesh.radius} m")

def main():
    """Main demonstration function"""
    
    print("SUAVE Vehicle STL Export Demonstration")
    print("="*50)
    
    # Create a simple aircraft
    print("Creating simple aircraft...")
    vehicle = create_simple_aircraft()
    print(f"Vehicle created: {vehicle.tag}")
    print(f"Components: {len(vehicle.wings)} wings, {len(vehicle.fuselages)} fuselages")
    
    # Demonstrate component-specific settings
    demonstrate_component_specific_settings(vehicle)
    
    # Export with different mesh qualities
    demonstrate_mesh_quality_comparison(vehicle)
    
    print("\n" + "="*60)
    print("EXPORT COMPLETE")
    print("="*60)
    print("Generated files:")
    print("- aircraft_coarse.stl, aircraft_coarse.key")
    print("- aircraft_medium.stl, aircraft_medium.key") 
    print("- aircraft_fine.stl, aircraft_fine.key")
    print("- aircraft_coarse.vsp3, aircraft_medium.vsp3, aircraft_fine.vsp3")
    
    print("\nTo view the STL files, you can use:")
    print("- MeshLab: Open the .stl files")
    print("- ParaView: Open the .stl files")
    print("- OpenVSP: Open the .vsp3 files")

if __name__ == "__main__":
    main()