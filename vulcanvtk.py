#!python
# Copyright 2019 Vale
# v2.0 11/2021 paulo.ernesto
# v1.0 05/2019 paulo.ernesto

import numpy as np
from skimage.transform import matrix_transform, AffineTransform
import pyvista as pv

# moved to pd_vtk vtk_Voxel.add_arrays_from_bmf
'''
def bm_to_vtk(bm, condition, variables):
  
  from pd_vtk import vtk_Voxel
  grid = vtk_Voxel.from_bmf(bm)
  if isinstance(variables, str):
    variables = [variables]
  # its easy to make a UniformGrid, but we will need functions
  # only available to a StructuredGrid
  grid = grid.cast_to_structured_grid()

  cv = [np.ndarray(grid.GetNumberOfCells(), dtype=[np.object, np.float_, np.float_, np.int_, np.int_][['name', 'integer', '***', 'float', 'bool'].index(bm.field_type(v))]) for v in variables]

  bl = None
  if condition:
    block_select = bm_sanitize_condition(condition)
    bl = bm.get_matches(block_select)
  
  cells = grid.cell_centers().points
  for cellId in range(grid.GetNumberOfCells()):
    #xyz = VoxelVTK.sGetCellCenter(grid, cellId)
    xyz = cells[cellId]
    #print(xyz)
    if bm.find_xyz(*xyz):
      # point outside block model data region
      grid.BlankCell(cellId)
      #print("blank1")
    elif bl is not None and bm.get_position() not in bl:
      grid.BlankCell(cellId)
      #print("blank2")
    else:
      #print("flag1")
      for i in range(len(variables)):
        # if bm.is_string(variables[i]):
        if cv[i].dtype == np.object:
          cv[i][cellId] = bm.get_string(variables[i])
        else:
          cv[i][cellId] = bm.get(variables[i])
        print(xyz,cellId,i,cv[i][cellId])
      
  for i in range(len(variables)):
    grid.cell_arrays[variables[i]] = cv[i]

  return grid
'''
def vtri_to_vtk(input_path, input_scd = None):
  from vulcan_mapfile import VulcanScd
  from vulcan_save_tri import vulcan_load_tri
  from pd_vtk import vtk_nf_to_mesh
  nodes, faces, cv, cn = vulcan_load_tri(input_path)
  color = [0, 0, 0, 255]
  if cn == 'rgb':
    d = [2**24, 2**16, 2**8, 1]
    # print(cv)
    # print(np.roll(d, 1))
    rgb256 = np.mod(cv, d[:-1])
    # print(rgb256)
    rgb256 = np.divide(np.mod(cv, d[:-1]), d[1:]).astype(np.int_)
    # print(rgb256)
    color = np.divide(rgb256, 255).tolist()
    # print(color)

  imesh = vtk_nf_to_mesh(nodes, faces)
  if input_scd is None:
    return imesh, None
  else:
    if cn == 'colour':
      scd = VulcanScd(input_scd)
      color = scd.index_to_rgb(cv)
    return imesh, color

def create_scd_colormap(bm, variable, scd):
  from matplotlib.colors import ListedColormap
  tv = bm.get_translation_values(variable)
  ci = np.ones((len(tv), 4))
  for i in range(len(tv)):
    rgb = scd.get_rgb(tv[i], True)
    if rgb is not None:
      ci[i] = rgb
  lcm = ListedColormap(ci)
  return lcm
