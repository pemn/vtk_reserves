# vtk_reserves
ore body reserve report using geological block model, topographical surfaces and region solids
  
## Description
This is a complete standalone toolchain to calculate ore body reserves from a geological block model, topographical surfaces and region solids.  
Its intended for high level use, with no required knowledge of programming or python.  
While there are many software packages that enable this calculation, this is a fully open source sollution. Free, auditable and extendable.  
It integrates multiple mature python modules to handle each step of this complex process.  
The modules used include:  
 - pyvista (geometry operations and visual rendering)
 - pandas (number crunching)
 - tkinter (graphical interface)

## Maturity
This solution is feature complete but still under active development.  
Its intended for academic purposes, reserve comparison with mature tools and as a technological demonstrator of solving the required steps.  
Its not yet suited for production use.  
## How to install
First, install a python distribution such as [WinPython](https://winpython.github.io/).  
Download all files from this repository root to a local folder with execute permissions. On Windows this means a folder outside the user directories, because locations such as Downloads, Documents, Desktop, etc do not allow .cmd files to run.  
Only one required module is not included by default in WinPython and other mainstrean distros: pyvista. Install using:  
`pip install pyvista`
## Run
The simplest way to run is to execute (double click on windows explorer) the supplied vtk_reserves.cmd file. This batch script shoud detect a WinPython distribution and use it automatically to call the main py file (vtk_reserves.py).  
For other distributions, manually call the main script with:  
`python vtk_reserves.py`  
Either way the user interface should appear:  
![screenshot1](./assets/screenshot1.png?raw=true)
## Preparing Data
The included converter tool bm_to_vtk.py can be used to convert data to the supported formats.
For instance, it can convert Vulcan block models (BMF) to the VTK format.
It can also convert between multiple supported triangulation formats. See next section.
## Usage
The block model must be in one of the supported formats:
 - csv file, containing x,y,z of each block and variables.
 - vtk file with a UniformGrid object type and the variables as cell_arrays.  

The surfaces and solids can be in those supported formats:
 - csv (ASCII, with x,y,z and faces)
 - obj (wavefront)
 - msh (leapfrog)
 - vtk (containing PolyData Mesh object types).  

Regions are used to split the report by the areas within closed solids. Each area/region will have the same name as the file.   
Mine exclude and mine include are used to "mine up" or "mine down". Usualy topographies are "mine up" (include) and mining pit are "mine down" (exclude).  
Output can be in one of those formats:
 - csv
 - xlsx
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
  
If the display checkbox is ticked, a 3d window should appear showing the data used:  
![screenshot2](./assets/screenshot2.png?raw=true)
## Sample Data
There is a simple artificial dataset on the sample_data folder of this repository for testing and reference.
## License
Apache 2.0

