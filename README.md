# vtk_reserves
ore body reserve calculation using block model, surface and solids
  
## Description
This is a complete toolchain to calculate ore body reserves from a geological block model, topographical surfaces and region solids.  
While there are many software packages that enable this calculation, this is a fully open source sollution. Free, auditable and extendable.  
It integrates multiple mature python modules to handle each step of this complex process.  
The mudules used include:  
 - pyvista (geometry calculations and rendering
 - pandas (number crunching)
 - tkinter (graphical interface)

## Maturity
This solution is under active development.  
## How to install
Install a python distribution such as WinPython.  
Download the files in the root folder of this repository to a folder with execute permissions. On Windows this means a folder outside the user directories, because locations such as Downloads, Documents, Desktop, etc do not allow .cmd files to run.  
## Run
The simples way to run is to execute the supplied .cmd file, which should detect a WinPython distribution and use it.  
For other distributions, manually call the main script with:
`python vtk_reserves.py`
Either way the user interface should appear:  
![screenshot1](./assets/screenshot1.png?raw=true)
## Output
The result is a reserves report split by lithology, region and possibly other classificatory variables.  
It will also contain volume, mass and weighted mean of numerical variables such as grade.
| region | lito | grade mean | density mean | volume sum | mass
| --- | --- | --- | --- | --- | --- |
| vox_region1 | high |  |  | 0.0 | 0.0 |
| vox_region1 | low |  |  | 0.0 | 0.0 |
| vox_region1 | medium |  |  | 0.0 | 0.0 |
| vox_region2 | high | 69.53446163437245 | 87.6740902065477 | 28468000.0 | 2495906000.0 |
| vox_region2 | low | 21.11175785797439 | 79.5 | 17180000.0 | 1365810000.0 |
| vox_region2 | medium | 45.963415442028904 | 81.37556154537286 | 44520000.0 | 3622840000.0 |
| vox_region3 | high | 69.4 | 84.5 | 7500000.0 | 633750000.0 |
| vox_region3 | low | 21.64715704429221 | 78.52418911489829 | 7276000.0 | 571342000.0 |
| vox_region3 | medium | 44.76301545763373 | 77.48488664987406 | 15880000.0 | 1230460000.0 |
  
![screenshot2](./assets/screenshot2.png?raw=true)
## Sample Data
There is a simple artificial dataset on the sample_data folder of this repository
## File Formats
The block model can be a csv or vtk. The vtk object type should be a UniformGrid with the block model variables as cell_arrays.  
A block model conversor from Vulcan file format (BMF) is included: bm_to_vtk.py.  
The surfaces and solids can be csv, obj (wavefront), msh (leapfrog) or vtk. The bm_to_vtk.py will also convert between this formats, and also from the 00t Vulcan format. 
## License
Apache 2.0

