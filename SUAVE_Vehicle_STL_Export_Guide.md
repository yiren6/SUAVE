# SUAVE Vehicle Definition and STL Export Guide

This guide demonstrates how to use SUAVE to define aircraft vehicles and export their STL surface meshes.

## Overview

SUAVE (Stanford University Aerospace Vehicle Environment) is a framework for aircraft design and analysis. It provides capabilities to:

1. **Define Aircraft Geometry**: Create detailed vehicle models with wings, fuselages, engines, and other components
2. **Export to OpenVSP**: Convert SUAVE vehicle definitions to OpenVSP format
3. **Generate STL Meshes**: Create surface mesh files for 3D visualization and analysis

## Files in this Example

- `simple_suave_vehicle_example.py` - Basic vehicle definition without external dependencies
- `suave_vehicle_stl_export_example.py` - Complete example with STL export functionality
- `SUAVE_Vehicle_STL_Export_Guide.md` - This documentation

## Quick Start

### 1. Basic Vehicle Definition

```python
import SUAVE
from SUAVE.Core import Units

# Create a vehicle
vehicle = SUAVE.Vehicle()
vehicle.tag = 'My_Aircraft'

# Set basic properties
vehicle.mass_properties.max_takeoff = 1200.0  # kg
vehicle.reference_area = 16.0  # m²
vehicle.passengers = 4

# Add a wing
wing = SUAVE.Components.Wings.Main_Wing()
wing.tag = 'main_wing'
wing.aspect_ratio = 8.0
wing.spans.projected = 11.0  # m
wing.chords.root = 2.0  # m
wing.chords.tip = 1.2  # m
wing.areas.reference = 16.0  # m²

vehicle.append_component(wing)
```

### 2. STL Export (Requires OpenVSP)

```python
from SUAVE.Input_Output.OpenVSP import write as vsp_write
from SUAVE.Input_Output.OpenVSP import write_vsp_mesh

# Export to OpenVSP format
vsp_write(vehicle, 'my_aircraft')

# Generate STL mesh
write_vsp_mesh(vehicle, 'my_aircraft', 
               half_mesh_flag=True,    # Create half mesh with symmetry
               growth_ratio=1.2,       # Mesh growth ratio
               growth_limiting_flag=True)  # Use 3D growth limiting
```

## Vehicle Components

### Wings

SUAVE supports various wing types:

```python
# Main wing
wing = SUAVE.Components.Wings.Main_Wing()
wing.tag = 'main_wing'
wing.aspect_ratio = 8.0
wing.sweeps.quarter_chord = 0.0 * Units.deg
wing.thickness_to_chord = 0.12
wing.taper = 0.6
wing.dihedral = 2.0 * Units.deg

# Dimensions
wing.spans.projected = 11.0  # m
wing.chords.root = 2.0  # m
wing.chords.tip = 1.2  # m
wing.areas.reference = 16.0  # m²

# Position
wing.origin = [[2.0, 0.0, 0.0]]  # [x, y, z] in meters
wing.aerodynamic_center = [0.0, 0.0, 0.0]

# Properties
wing.vertical = False
wing.symmetric = True
wing.high_lift = True
```

### Fuselage

```python
fuselage = SUAVE.Components.Fuselages.Fuselage()
fuselage.tag = 'fuselage'

# Dimensions
fuselage.lengths.total = 8.0  # m
fuselage.width = 1.2  # m
fuselage.heights.maximum = 1.4  # m
fuselage.effective_diameter = 1.1  # m

# Shape parameters
fuselage.nose_curvature = 1.5
fuselage.tail_curvature = 1.5
fuselage.fineness.nose = 0.8
fuselage.fineness.tail = 0.8

# Position
fuselage.origin = [[0.0, 0.0, 0.0]]
```

### Engines

```python
# Turbofan engine
turbofan = SUAVE.Components.Energy.Networks.Turbofan()
turbofan.tag = 'turbofan'
turbofan.number_of_engines = 1
turbofan.bypass_ratio = 5.0
turbofan.engine_length = 2.0  # m
turbofan.nacelle_diameter = 0.8  # m
turbofan.origin = [[6.0, 0.0, 0.0]]  # Position
```

## STL Export Process

### Prerequisites

1. **SUAVE**: Install SUAVE framework
2. **OpenVSP**: Install OpenVSP and its Python API
3. **Python Modules**: `vsp` or `openvsp` module must be available

### Export Steps

1. **Create Vehicle**: Define your aircraft in SUAVE
2. **Export to OpenVSP**: Use `vsp_write()` to create `.vsp3` file
3. **Generate STL**: Use `write_vsp_mesh()` to create `.stl` file

### Export Parameters

```python
write_vsp_mesh(vehicle, tag, half_mesh_flag, growth_ratio, growth_limiting_flag)
```

- `vehicle`: SUAVE vehicle object
- `tag`: Base filename (without extension)
- `half_mesh_flag`: Create half mesh with symmetry plane (True/False)
- `growth_ratio`: Mesh growth ratio (typically 1.1-1.3)
- `growth_limiting_flag`: Use 3D growth limiting (True/False)

### Output Files

- `{tag}.vsp3` - OpenVSP geometry file
- `{tag}.stl` - STL surface mesh
- `{tag}.key` - Surface identification file

## Advanced Features

### Wing Segments

For complex wing geometries, use segments:

```python
# Create wing segments
segment = SUAVE.Components.Wings.Segment()
segment.tag = 'Root'
segment.percent_span_location = 0.0
segment.twist = 2.0 * Units.deg
segment.root_chord_percent = 1.0
segment.thickness_to_chord = 0.12
segment.dihedral_outboard = 2.5 * Units.degrees
segment.sweeps.quarter_chord = 28.225 * Units.degrees

wing.append_segment(segment)
```

### Control Surfaces

```python
# Add flaps
flap = SUAVE.Components.Wings.Control_Surfaces.Flap()
flap.tag = 'flap'
flap.span_fraction_start = 0.15
flap.span_fraction_end = 0.324
flap.deflection = 1.0 * Units.deg
flap.chord_fraction = 0.19
wing.append_control_surface(flap)
```

### Airfoils

```python
# Define airfoil
airfoil = SUAVE.Components.Airfoils.Airfoil()
airfoil.coordinate_file = 'path/to/airfoil.txt'
segment.append_airfoil(airfoil)
```

## Troubleshooting

### Common Issues

1. **OpenVSP Import Error**: 
   - Install OpenVSP with Python API
   - Ensure `vsp` or `openvsp` module is available

2. **STL Generation Fails**:
   - Check OpenVSP installation
   - Verify vehicle geometry is valid
   - Try different mesh parameters

3. **Geometry Issues**:
   - Ensure all dimensions are positive
   - Check that components don't overlap
   - Verify coordinate systems

### Error Handling

```python
try:
    from SUAVE.Input_Output.OpenVSP import write as vsp_write
    vsp_write(vehicle, 'my_aircraft')
    print("✓ OpenVSP export successful")
except ImportError:
    print("⚠ OpenVSP not available")
except Exception as e:
    print("⚠ Export failed: {}".format(str(e)))
```

## Example Usage

Run the example scripts:

```bash
# Basic vehicle definition (no external dependencies)
python simple_suave_vehicle_example.py

# Complete example with STL export (requires OpenVSP)
python suave_vehicle_stl_export_example.py
```

## Further Reading

- [SUAVE Documentation](https://suave.stanford.edu/)
- [OpenVSP Documentation](https://openvsp.org/)
- [SUAVE GitHub Repository](https://github.com/suavecode/SUAVE)

## License

This example code is provided for educational purposes. Please refer to SUAVE's license for usage terms.