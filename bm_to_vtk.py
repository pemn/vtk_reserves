#!python
# convert a vulcan data to VTK format
# v1.0 09/2020 paulo.ernesto
# Copyright 2020 Vale
# License: Apache 2.0
"""
usage: $0 input_data*bmf,00t,vtk,vti condition variables#variable:input_data output_vtk*vtk,vti display@
"""

import sys, os.path

# import modules from a pyz (zip) file with same name as scripts
sys.path.insert(0, os.path.splitext(sys.argv[0])[0] + '.pyz')

from _gui import pyd_zip_extract, usage_gui, bm_sanitize_condition, commalist

import numpy as np
import pandas as pd

pyd_zip_extract()
import pyvista as pv
from vulcanvtk import vtri_to_vtk
from pd_vtk import vtk_Voxel

# convert a vulcan surface to a vulcan solid
def main(input_data, condition, variables, output_vtk, display):
  mesh = None
  color = None
  if not variables:
    variables = []
  else:
    variables = commalist().parse(variables).split()
  print("variables")
  print(variables)
  if input_data.lower().endswith('bmf'):
    import vulcan
    if len(variables) == 0:
      variables = ['volume']
    bm = vulcan.block_model(input_data)
    #mesh = bm_to_vtk(bm, condition, variables)
    grid = vtk_Voxel.from_bmf(bm)
    mesh = grid.add_arrays_from_bmf(bm, condition, variables)
  elif input_data.lower().endswith('00t'):
    mesh, color = vtri_to_vtk(input_data)
  else:
    mesh = pv.read(input_data)

  if output_vtk:
    mesh.save(output_vtk)

  if int(display):
    p = pv.Plotter()
    if color is None:
      p.add_mesh(mesh)
    else:
      p.add_mesh(mesh, color=color)
    p.show()

  print("finished")

if __name__=="__main__":
  usage_gui(__doc__)
