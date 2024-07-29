#!python
# mine block inside/outside solids or up/below surfaces
# blocks: block model, grid or point cloud with block to be mined
# mine_include: blocks outside/above are mined (mine = 0)
# mine_exclude: blocks inside/below are mined (mine = 0)
# output: path to save calcuted grid

'''
usage: $0 blocks*vtk,csv,xlsx mine_include#mesh_include*vtk,obj,msh mine_exclude#mesh_exclude*vtk,obj,msh mine=mine output*vtk,csv,xlsx display@
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

import sys, os.path
import numpy as np
import pandas as pd
import re

# import modules from a pyz (zip) file with same name as scripts
sys.path.insert(0, os.path.splitext(sys.argv[0])[0] + '.pyz')

from _gui import commalist, usage_gui, commalist, pd_load_dataframe, pd_save_dataframe, log, pyd_zip_extract

pyd_zip_extract()

from pd_vtk import pv_read, pv_save, vtk_df_to_mesh, vtk_mesh_to_df, vtk_plot_meshes, vtk_Voxel, vtk_meshes_bb, mr_block_mine, vtk_block_mine

class GridMine(object):
  _mine = 'mine'
  _grid = None
  _blank = True
  _m0 = []
  _m1 = []
  _gz = None
  def __init__(self, grid = None, mine = None):
    if grid:
      self.set_grid(grid)
    if mine:
      self._mine = mine

  def set_grid(self, grid):
    self._grid = grid
    #self._gz = np.zeros(grid.n_cells)
    #self._gz = np.full(grid.n_cells, np.inf)
    self._gz = np.zeros(self._grid.n_cells, dtype=np.ubyte)

  def fill(self, value):
    self._gz.fill(value)
    self._blank = False

  @property
  def blank(self):
    return self._blank

  @property
  def meshes(self):
    if self._grid:
      return self._m0 + self._m1 + [self._grid]
    return self._m0 + self._m1

  def mine_include(self, mesh):
    self._m0.append(mesh)

  def mine_exclude(self, mesh):
    self._m1.append(mesh)

  def calc_mine(self):
    self._gz = vtk_block_mine(self._m0, self._grid)
    
    if self.blank:
      self.fill(np.inf)

    mine = vtk_block_mine(self._m1, self._grid)
    self._gz = np.multiply(self._gz, np.where(np.isnan(mine), 1.0, np.subtract(1.0, mine)))

  
  def __call__(self):
    if self._grid:
      self.calc_mine()
      self._grid.cell_data[self._mine] = np.asfarray(self._gz)
    return self._grid


def vtk_mine(blocks, mine_include, mine_exclude, mine, output, display):

  gm = GridMine(None, mine)

  for fp in commalist().parse(mine_include).split():
    if os.path.exists(fp):
      mesh = pv_read(fp)
      gm.mine_include(mesh)

  for fp in commalist().parse(mine_exclude).split():
    if os.path.exists(fp):
      mesh = pv_read(fp)
      gm.mine_exclude(mesh)

  if re.fullmatch(r'[\d\.\-,;_~]+', blocks):
    bb = vtk_meshes_bb(gm.meshes)
    grid = vtk_Voxel.from_bb_schema(bb, blocks)
    grid.cells_volume('volume')
  else:
    grid = vtk_Voxel.from_file_path(blocks)
  gm.set_grid(grid)

  gm()

  if output:
    pv_save(grid, output)

  if int(display):
    vtk_plot_meshes(gm.meshes)
  log("# vtk_mine finished")

main = vtk_mine

if __name__=="__main__":
  usage_gui(__doc__)
