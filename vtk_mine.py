#!python
# mine block inside/outside solids or up/below surfaces
# blocks: block model, grid or point cloud with block to be mined
# mine_include: blocks outside/above are mined (mine = 0)
# mine_exclude: blocks inside/below are mined (mine = 0)
# output: path to save calcuted grid

'''
usage: $0 blocks*vtk,csv,xlsx mine_include#mesh_include*vtk,obj,msh mine_exclude#mesh_exclude*vtk,obj,msh output*vtk,csv,xlsx display@
'''
'''
Copyright 2017 - 2021 Vale

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

import sys
import numpy as np
import pandas as pd
import re

import os.path

# import modules from a pyz (zip) file with same name as scripts
sys.path.insert(0, os.path.splitext(sys.argv[0])[0] + '.pyz')

from _gui import commalist, usage_gui, commalist, pd_load_dataframe, pd_save_dataframe

from pd_vtk import pv_read, pv_save, vtk_df_to_mesh, vtk_mesh_to_df, vtk_plot_meshes, vtk_Voxel

class GridMine(object):
  def __init__(self, grid):
    self._grid = grid
    #self._gz = np.zeros(grid.n_cells)
    #self._gz = np.full(grid.n_cells, np.inf)
    self._gz = np.zeros(self._grid.n_cells, dtype=np.bool)
    self._blank = True

  def fill(self, value):
    #if self._gz is None:
    #  self._gz = np.full(self._grid.n_cells, value, dtype=np.float)
    #else:
    self._gz.fill(value)
    self._blank = False

  @property
  def blank(self):
    return self._blank

  def mine_include(self, mesh):
    return self.mine_mesh(mesh, False)

  def mine_exclude(self, mesh):
    return self.mine_mesh(mesh, True)

  def mine_mesh(self, mesh, out = False):
    self._blank = False
    # StructuredGrid.select_enclosed_points(surface, tolerance=0.001, inside_out=False, check_surface=True, progress_bar=False)
    #mg = self._grid.select_enclosed_points(mesh, check_surface=False)
    mg = self._grid.compute_implicit_distance(mesh)
    
    mz = mg.ptc().get_array('implicit_distance')
    if out:
      #mz *= -1
      mz = np.less(mz, 0)
    else:
      mz = np.greater_equal(mz, 0)
    # TODO: compute_normals
    #if self._gz is None:
    #  self._gz = mz
    if out:
      #print(np.min([self._gz, mz], 0))
      #self._gz = np.min([self._gz, mz], 0)
      self._gz &= mz
    else:
      #self._gz = np.max([self._gz, mz], 0)
      self._gz |= mz

    return None
  
  def get(self, mine='mine'):
    self._grid.cell_arrays[mine] = self._gz.astype(np.float)
    #self._grid.set_active_scalars(mine)
    return self._grid

def vtk_mine(blocks, mine_include, mine_exclude, output, display):
  print("main")
  grid = vtk_Voxel.from_file_path(blocks)
  print(grid)
  meshes = []
  gm = GridMine(grid)

  for fp in commalist().parse(mine_include).split():
    if os.path.exists(fp):
      mesh = pv_read(fp)
      meshes.append(mesh)
      gm.mine_include(mesh)

  if gm.blank:
    #mm.fill(np.NINF)
    gm.fill(np.inf)

  for fp in commalist().parse(mine_exclude).split():
    if os.path.exists(fp):
      mesh = pv_read(fp)
      meshes.append(mesh)
      gm.mine_exclude(mesh)

  meshes.append(gm.get())

  if output:
    pv_save(grid, output)

  if int(display):
    vtk_plot_meshes(meshes)

main = vtk_mine

if __name__=="__main__":
  usage_gui(__doc__)
