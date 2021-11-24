#!python
# convert a vulcan data to VTK format
# v1.0 09/2020 paulo.ernesto
# Copyright 2020 Vale
# License: Apache 2.0
"""
usage: $0 input_data*bmf,00t,vtk,vti condition variables#variable:input_data output*obj,vtk,vti display@
"""

import sys, os.path

# import modules from a pyz (zip) file with same name as scripts
sys.path.insert(0, os.path.splitext(sys.argv[0])[0] + '.pyz')

from _gui import pyd_zip_extract, usage_gui, bm_sanitize_condition, commalist, wavefront_save_obj, table_name_selector

import numpy as np
import pandas as pd

pyd_zip_extract()
import pyvista as pv
#from vulcanvtk import vtri_to_vtk
from vulcan_save_tri import vulcan_load_tri
from pd_vtk import vtk_Voxel, vtk_meshes_to_obj, vtk_plot_meshes, pv_save

# convert a vulcan surface to a vulcan solid
def main(input_data, condition, variables, output, display):
  mesh = None
  color = None
  if not variables:
    variables = []
  else:
    variables = commalist().parse(variables).split()
  print("variables")
  print(variables)
  file_path, table_name = table_name_selector(input_data)

  if file_path.lower().endswith('bmf'):
    import vulcan
    if len(variables) == 0:
      variables = ['volume']
    
    bm = vulcan.block_model(file_path)

    grid = vtk_Voxel.from_bmf(bm, table_name)
    mesh = grid.add_arrays_from_bmf(bm, condition, variables)
  elif file_path.lower().endswith('00t'):
    nodes, faces, cv, cn = vulcan_load_tri(input_path)
    mesh = vtk_nf_to_mesh(nodes, faces)
  else:
    mesh = pv.read(file_path)
  print(mesh)
  if output.lower().endswith('obj'):
    od = vtk_meshes_to_obj([mesh])
    wavefront_save_obj(output, od)
  elif output:
    pv_save(mesh, output)

  if int(display):
    vtk_plot_meshes([mesh])

  print("finished")

if __name__=="__main__":
  usage_gui(__doc__)
