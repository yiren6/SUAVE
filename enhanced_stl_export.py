#!/usr/bin/env python3
"""
Enhanced STL Export Module for SUAVE Vehicles

This module provides improved STL export capabilities with:
- Direct vehicle-to-STL export
- Configurable mesh quality settings
- Support for multiple components
- Mesh validation and statistics
- Error handling and logging

Author: AI Assistant
Date: 2024
"""

import sys
import os
import numpy as np
import time
from copy import deepcopy

# Add SUAVE to path
sys.path.append('/workspace/trunk')

import SUAVE
from SUAVE.Core import Units, Data
from SUAVE.Input_Output.OpenVSP.write_vsp_mesh import write_vsp_mesh, set_sources
from SUAVE.Input_Output.OpenVSP.vsp_write import write as write_vsp

try:
    import vsp as vsp
except ImportError:
    try:
        import openvsp as vsp
    except ImportError:
        vsp = None

class STLExportError(Exception):
    """Custom exception for STL export errors"""
    pass

class MeshQualityPresets:
    """Predefined mesh quality settings"""
    
    COARSE = {
        'elements_per_chord': 25,
        'growth_ratio': 1.3,
        'growth_limiting': False,
        'half_mesh': True,
        'description': 'Fast generation, lower quality'
    }
    
    MEDIUM = {
        'elements_per_chord': 50,
        'growth_ratio': 1.2,
        'growth_limiting': True,
        'half_mesh': True,
        'description': 'Balanced quality and speed'
    }
    
    FINE = {
        'elements_per_chord': 100,
        'growth_ratio': 1.1,
        'growth_limiting': True,
        'half_mesh': True,
        'description': 'High quality, slower generation'
    }
    
    ULTRA_FINE = {
        'elements_per_chord': 200,
        'growth_ratio': 1.05,
        'growth_limiting': True,
        'half_mesh': True,
        'description': 'Maximum quality, very slow'
    }

class EnhancedSTLExporter:
    """Enhanced STL exporter with advanced features"""
    
    def __init__(self, verbose=True):
        """
        Initialize the STL exporter
        
        Parameters:
        -----------
        verbose : bool
            Enable verbose output
        """
        self.verbose = verbose
        self.export_history = []
        
        if vsp is None:
            raise STLExportError("OpenVSP Python API not available. Please install OpenVSP.")
    
    def log(self, message):
        """Log message if verbose mode is enabled"""
        if self.verbose:
            print(f"[STL Export] {message}")
    
    def export_vehicle_stl(self, vehicle, output_file, mesh_quality='medium', 
                          include_components=None, custom_settings=None):
        """
        Export SUAVE vehicle to STL format with enhanced options
        
        Parameters:
        -----------
        vehicle : SUAVE.Vehicle
            SUAVE vehicle object
        output_file : str
            Output filename (without extension)
        mesh_quality : str or dict
            Mesh quality preset ('coarse', 'medium', 'fine', 'ultra_fine') or custom dict
        include_components : list, optional
            Components to include ['wings', 'fuselages', 'nacelles']
        custom_settings : dict, optional
            Custom mesh settings to override defaults
            
        Returns:
        --------
        dict : Export results including file paths and statistics
        """
        
        start_time = time.time()
        
        try:
            # Validate inputs
            self._validate_inputs(vehicle, output_file)
            
            # Get mesh settings
            settings = self._get_mesh_settings(mesh_quality, custom_settings)
            
            # Filter components if specified
            filtered_vehicle = self._filter_components(vehicle, include_components)
            
            self.log(f"Starting STL export for vehicle '{vehicle.tag}'")
            self.log(f"Output file: {output_file}")
            self.log(f"Mesh quality: {settings.get('description', 'custom')}")
            
            # Step 1: Export to OpenVSP format
            self.log("Converting vehicle to OpenVSP format...")
            vsp.ClearVSPModel()
            write_vsp(filtered_vehicle, output_file, verbose=self.verbose, write_file=True)
            
            # Step 2: Generate STL mesh
            self.log("Generating STL surface mesh...")
            self._generate_stl_mesh(filtered_vehicle, output_file, settings)
            
            # Step 3: Validate and get statistics
            self.log("Validating generated mesh...")
            stats = self._get_mesh_statistics(output_file)
            
            # Record export history
            export_info = {
                'vehicle_tag': vehicle.tag,
                'output_file': output_file,
                'mesh_quality': mesh_quality,
                'settings': settings,
                'statistics': stats,
                'duration': time.time() - start_time,
                'timestamp': time.time()
            }
            self.export_history.append(export_info)
            
            self.log(f"STL export completed successfully in {export_info['duration']:.2f} seconds")
            
            return export_info
            
        except Exception as e:
            error_msg = f"STL export failed: {str(e)}"
            self.log(error_msg)
            raise STLExportError(error_msg) from e
    
    def _validate_inputs(self, vehicle, output_file):
        """Validate input parameters"""
        if not isinstance(vehicle, SUAVE.Vehicle):
            raise STLExportError("vehicle must be a SUAVE.Vehicle object")
        
        if not output_file or not isinstance(output_file, str):
            raise STLExportError("output_file must be a non-empty string")
        
        if not hasattr(vehicle, 'wings') and not hasattr(vehicle, 'fuselages'):
            raise STLExportError("Vehicle must have at least wings or fuselages")
    
    def _get_mesh_settings(self, mesh_quality, custom_settings):
        """Get mesh settings from preset or custom configuration"""
        
        # Get preset settings
        if isinstance(mesh_quality, str):
            preset_map = {
                'coarse': MeshQualityPresets.COARSE,
                'medium': MeshQualityPresets.MEDIUM,
                'fine': MeshQualityPresets.FINE,
                'ultra_fine': MeshQualityPresets.ULTRA_FINE
            }
            
            if mesh_quality not in preset_map:
                raise STLExportError(f"Unknown mesh quality preset: {mesh_quality}")
            
            settings = preset_map[mesh_quality].copy()
        else:
            settings = mesh_quality.copy()
        
        # Apply custom settings overrides
        if custom_settings:
            settings.update(custom_settings)
        
        return settings
    
    def _filter_components(self, vehicle, include_components):
        """Filter vehicle components based on include_components list"""
        if include_components is None:
            return vehicle
        
        filtered_vehicle = deepcopy(vehicle)
        
        # Filter wings
        if 'wings' not in include_components:
            filtered_vehicle.wings = SUAVE.Core.Container()
        
        # Filter fuselages
        if 'fuselages' not in include_components:
            filtered_vehicle.fuselages = SUAVE.Core.Container()
        
        # Filter nacelles
        if 'nacelles' not in include_components:
            filtered_vehicle.nacelles = SUAVE.Core.Container()
        
        return filtered_vehicle
    
    def _generate_stl_mesh(self, vehicle, output_file, settings):
        """Generate STL mesh using OpenVSP"""
        
        # Read the VSP file
        vsp.ReadVSPFile(output_file + '.vsp3')
        
        # Set output file types
        file_type = vsp.CFD_STL_TYPE + vsp.CFD_KEY_TYPE
        set_int = vsp.SET_ALL
        
        vsp.SetComputationFileName(vsp.CFD_STL_TYPE, output_file + '.stl')
        vsp.SetComputationFileName(vsp.CFD_KEY_TYPE, output_file + '.key')
        
        # Set mesh parameters
        self._configure_mesh_parameters(vehicle, settings)
        
        # Set sources for mesh refinement
        set_sources(vehicle)
        
        # Update and generate mesh
        vsp.Update()
        vsp.ComputeCFDMesh(set_int, file_type)
    
    def _configure_mesh_parameters(self, vehicle, settings):
        """Configure OpenVSP mesh parameters"""
        
        # Set multi-solid STL output
        vehicle_cont = vsp.FindContainer('Vehicle', 0)
        STL_multi = vsp.FindParm(vehicle_cont, 'MultiSolid', 'STLSettings')
        vsp.SetParmVal(STL_multi, 1.0)
        
        # Set far field parameters
        vsp.SetCFDMeshVal(vsp.CFD_FAR_FIELD_FLAG, 1)
        
        if settings.get('half_mesh', True):
            vsp.SetCFDMeshVal(vsp.CFD_HALF_MESH_FLAG, 1)
        
        # Calculate far field size
        vehicle_id = vsp.FindContainersWithName('Vehicle')[0]
        xlen = vsp.GetParmVal(vsp.FindParm(vehicle_id, "X_Len", "BBox"))
        ylen = vsp.GetParmVal(vsp.FindParm(vehicle_id, "Y_Len", "BBox"))
        zlen = vsp.GetParmVal(vsp.FindParm(vehicle_id, "Z_Len", "BBox"))
        
        max_len = np.max([xlen, ylen, zlen])
        far_length = 10.0 * max_len
        
        vsp.SetCFDMeshVal(vsp.CFD_FAR_SIZE_ABS_FLAG, 1)
        vsp.SetCFDMeshVal(vsp.CFD_FAR_LENGTH, far_length)
        vsp.SetCFDMeshVal(vsp.CFD_FAR_WIDTH, far_length)
        vsp.SetCFDMeshVal(vsp.CFD_FAR_HEIGHT, far_length)
        vsp.SetCFDMeshVal(vsp.CFD_FAR_MAX_EDGE_LEN, max_len)
        
        # Set mesh quality parameters
        vsp.SetCFDMeshVal(vsp.CFD_GROWTH_RATIO, settings.get('growth_ratio', 1.2))
        
        if settings.get('growth_limiting', True):
            vsp.SetCFDMeshVal(vsp.CFD_LIMIT_GROWTH_FLAG, 1.0)
        
        # Set element size based on chord length
        if hasattr(vehicle, 'wings') and len(vehicle.wings) > 0:
            # Use main wing MAC for sizing
            main_wing = vehicle.wings.main_wing if hasattr(vehicle.wings, 'main_wing') else vehicle.wings[0]
            MAC = main_wing.chords.mean_aerodynamic
            elements_per_chord = settings.get('elements_per_chord', 50)
            min_len = MAC / elements_per_chord
            vsp.SetCFDMeshVal(vsp.CFD_MAX_EDGE_LEN, min_len)
    
    def _get_mesh_statistics(self, output_file):
        """Get statistics about the generated mesh"""
        stl_file = output_file + '.stl'
        
        if not os.path.exists(stl_file):
            return {'error': 'STL file not found'}
        
        try:
            # Basic file statistics
            file_size = os.path.getsize(stl_file)
            
            # Try to read STL file for more detailed statistics
            stats = {
                'file_size_bytes': file_size,
                'file_size_mb': file_size / (1024 * 1024),
                'stl_file': stl_file,
                'key_file': output_file + '.key'
            }
            
            # Attempt to count triangles (basic STL parsing)
            try:
                with open(stl_file, 'rb') as f:
                    content = f.read()
                    # Count "facet normal" occurrences (rough triangle count)
                    triangle_count = content.count(b'facet normal')
                    stats['triangle_count'] = triangle_count
            except:
                stats['triangle_count'] = 'unknown'
            
            return stats
            
        except Exception as e:
            return {'error': f'Failed to get statistics: {str(e)}'}
    
    def get_export_history(self):
        """Get history of all exports performed"""
        return self.export_history
    
    def export_multiple_qualities(self, vehicle, output_base, qualities=['coarse', 'medium', 'fine']):
        """Export vehicle with multiple mesh qualities"""
        
        results = {}
        
        for quality in qualities:
            output_file = f"{output_base}_{quality}"
            self.log(f"Exporting {quality} quality mesh...")
            
            try:
                result = self.export_vehicle_stl(vehicle, output_file, mesh_quality=quality)
                results[quality] = result
                self.log(f"✓ {quality} export completed")
            except Exception as e:
                self.log(f"✗ {quality} export failed: {str(e)}")
                results[quality] = {'error': str(e)}
        
        return results

def create_enhanced_example():
    """Create an example demonstrating enhanced STL export capabilities"""
    
    print("Enhanced STL Export Example")
    print("="*40)
    
    # Create exporter
    exporter = EnhancedSTLExporter(verbose=True)
    
    # Create a simple vehicle (reuse from previous example)
    from example_stl_export import create_simple_aircraft
    vehicle = create_simple_aircraft()
    
    # Export with different qualities
    print("\nExporting with multiple mesh qualities...")
    results = exporter.export_multiple_qualities(vehicle, "enhanced_aircraft")
    
    # Print results summary
    print("\nExport Results Summary:")
    print("-" * 30)
    for quality, result in results.items():
        if 'error' in result:
            print(f"{quality}: FAILED - {result['error']}")
        else:
            stats = result['statistics']
            print(f"{quality}: SUCCESS")
            print(f"  Duration: {result['duration']:.2f}s")
            print(f"  File size: {stats.get('file_size_mb', 'unknown'):.2f} MB")
            print(f"  Triangles: {stats.get('triangle_count', 'unknown')}")
    
    # Show export history
    print(f"\nTotal exports performed: {len(exporter.get_export_history())}")
    
    return results

if __name__ == "__main__":
    create_enhanced_example()