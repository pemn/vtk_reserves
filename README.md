# vtk_reserves
ore body reserve calculation using block model, surface and solids
  
## Description
This is a complete toolchain to calculate ore body reserves from a geological block model, topographical surfaces and region solids.  
While there are many software packages that enable this calculation, this is a fully open source sollution. Free, auditable and extendable.  
It integrates multiple mature, python modules to handle each step of this complex process.  
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
The result is a reserves table split by lithology, region and possibly other classificatory variables.  
It will also contain volume, mass and weighted mean of numerical variables such as grade.
## Sample Data
There is a simple artificial dataset on the sample_data folder of this repository
## License
Apache 2.0

