#!python
# flag solid regions in a block model

'''
usage: $0 block_model*vtk,csv,xlsx regions#region*vtk,dxf,dwg,msh,obj,00t flag_var=region flag2d@ output*vtk display@
'''
import sys
import pandas as pd
import numpy as np
import re

import os.path

# import modules from a pyz (zip) file with same name as scripts
sys.path.insert(0, os.path.splitext(sys.argv[0])[0] + '.pyz')
from _gui import usage_gui, commalist, pyd_zip_extract, pd_load_dataframe, pd_save_dataframe, df_to_nodes_faces
#pyd_zip_extract()

from pd_vtk import pv_read, vtk_dmbm_to_ug, vtk_df_to_mesh, vtk_mesh_to_df, vtk_cells_to_flat, vtk_plot_meshes, vtk_nf_to_mesh

def pd_string_to_index(df):
  ''' convert string columns to index to avoid errors in vtk '''
  for c in df.columns:
    if df[c].dtype.name == 'object':
      df[c] = pd.factorize(df[c])[0]
  return df

def vtk_flag_region_2d(grid, meshes, flag_var, values = None):
  cv = np.empty(grid.n_points, dtype='object')
  ms = []
  for n in range(len(meshes)):
    v = n + 1
    if n < len(values):
      v = values[n]
    mesh = meshes[n]
    bounds = mesh.bounds
    for i in range(grid.n_points):
      p0 = grid.points[i].copy()
      p1 = p0.copy()
      # create a line crossing the mesh bounding box in Z
      p0[2] = min(bounds[4], bounds[5]) - 1
      p1[2] = max(bounds[4], bounds[5]) + 1
      # check if the line hits the mesh anywhere
      ip, ic = mesh.ray_trace(p0, p1)
      #print(p0, p1, ip.size, ic.size)
      if ic.size:
        cv[i] = v


  grid.point_arrays[flag_var] = cv

  return grid

def vtk_flag_region(grid, meshes, flag_var, flag_cell = False, values = None):
  if flag_cell:
    cv = np.full(grid.GetNumberOfCells(), '', dtype=np.object)
  else:
    cv = np.full(grid.GetNumberOfPoints(), '', dtype=np.object)
  
  if values is None or not isinstance(values, list):
    values = []

  for n in range(len(meshes)):
    v = n + 1
    if n < len(values):
      v = values[n]
    
    r = grid.select_enclosed_points(meshes[n], check_surface=False)
    if flag_cell:
      rc = r.ptc().get_array('SelectedPoints')
    else:
      rc = r.get_array('SelectedPoints')
    cv[rc > 0] = v
  if flag_cell:
    grid.cell_arrays[flag_var] = cv
  else:
    grid.point_arrays[flag_var] = cv
  #grid.set_active_scalars(flag_var)
  return grid


def main(block_model, regions, flag_var, flag_2d, output, display):
  grid = None
  if re.search(r'vt.$', block_model, re.IGNORECASE):
    grid = pv_read(block_model)
  else:
    bdf = pd_load_dataframe(block_model)
    #bdf = pd_string_to_index(bdf)
    if set(['XC','YC','ZC']).issubset(bdf.columns):
      grid = vtk_dmbm_to_ug(bdf)
    else:
      grid = vtk_df_to_mesh(bdf)
  print(grid)
  
  meshes = []
  values = []
  for region in commalist().parse(regions).split():
    if not os.path.exists(region):
      print("file not found:",region)
      continue

    if re.search(r'vt.$', region, re.IGNORECASE):
      mesh = pv_read(region)
    else:
      mesh = vtk_nf_to_mesh(*df_to_nodes_faces(pd_load_dataframe(region)))
    values.append(os.path.splitext(os.path.basename(region))[0])
    # store the region names in a field array
    #mesh.add_field_array([name], flag_var)
    meshes.append(mesh)
  if len(meshes):
    if int(flag_2d):
      print("flag2d")
      vtk_flag_region_2d(grid, meshes, flag_var, values)
    else:
      print("flag3d")
      vtk_flag_region(grid, meshes, flag_var, False, values)
  #elif grid.IsA('vtkStructuredGrid') or grid.IsA('vtkUniformGrid'):
  #elif grid.IsA('vtkPolyData'):

  if re.search(r'vt.$', output, re.IGNORECASE):
    grid.save(output)
  else:
    df = vtk_mesh_to_df(grid)
    if output:
      pd_save_dataframe(df, output)
    else:
      print(df.to_string(index=False))

  if int(display):
    vtk_plot_meshes([grid] + meshes)

if __name__=="__main__":
  usage_gui(__doc__)
