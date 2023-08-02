#!python
# calculate ore reserves
# v1.1 11/2021 paulo.ernesto
# v1.0 10/2020 paulo.ernesto
'''
usage: $0 block_model*vtk,csv variables#variable:block_model#type=breakdown,count,sum,mean,min,max,var,std,sem,q1,q2,q3,p10,p90,major,list#weight:block_model regions#region*vtk,obj,msh mine_include#mesh_include*vtk,obj,msh mine_exclude#mesh_exclude*vtk,obj,msh output*xlsx display@
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

*** You can contribute to the main repository at: ***

https://github.com/pemn/vtk_reserves
---------------------------------
'''

import sys, os.path

# import modules from a pyz (zip) file with same name as scripts
sys.path.insert(0, os.path.splitext(sys.argv[0])[0] + '.pyz')

from _gui import usage_gui, pyd_zip_extract, commalist, pd_save_dataframe, log

import numpy as np
import pandas as pd
from pd_vtk import vtk_Voxel, pv_read, vtk_plot_meshes, vtk_mesh_info, vtk_mesh_to_df
from vtk_mine import GridMine
from vtk_flag_regions import vtk_flag_region
from bm_breakdown import pd_breakdown


def pd_grid_depletion(block_model, regions, mine_include, mine_exclude):
  ''' create raw dataframe with the reserves data '''
  meshes = []
  grid = vtk_Voxel.from_file_path(block_model)

  if 'volume' not in grid.array_names:
    grid.cells_volume('volume')

  gm = GridMine(grid)
  for fp in mine_include:
    if os.path.exists(fp):
      mesh = pv_read(fp)
      meshes.append(mesh)
      gm.mine_include(mesh)

  if gm.blank:
    gm.fill(1)
  for fp in mine_exclude:
    if os.path.exists(fp):
      mesh = pv_read(fp)
      meshes.append(mesh)
      gm.mine_exclude(mesh)

  grid = gm()

  r_meshes = []
  r_values = []
  for r_path in regions:
    if len(r_path) and os.path.exists(r_path):
      print(r_path)
      r_values.append(os.path.splitext(os.path.basename(r_path))[0])
      r_meshes.append(pv_read(r_path))
  meshes.extend(r_meshes)

  vtk_flag_region(grid, r_meshes, 'region', True, r_values)
  
  meshes.append(grid)

  #df = pd.DataFrame(np.transpose(grid.cell_data.values()), columns=grid.cell_data)
  
  df = vtk_mesh_to_df(grid)
  print(df.describe())
  if len(r_meshes):
    # exclude rows where region is empty
    df = df.query("region != ''")

  return df, meshes

def vl_add_weight(vl, w):
  for i in range(len(vl)):
    if len(vl[i]) > 1:
      if vl[i][1] in ['mean','sum']:
        if w not in vl[i]:
          vl[i].append(w)
  return vl

def vl_add_region(vl):
  for i in range(len(vl)):
    if len(vl[i]) > 1:
      if vl[i][0].startswith('region'):
        break
  else:
    vl.insert(0, ['region','',''])
  return vl

# convert a vulcan surface to a vulcan solid
def vtk_reserves(block_model, variables, regions, mine_include, mine_exclude, output, display):
  
  # ensure numeric variables are weighted by mine
  vl = vl_add_weight(commalist().parse(variables), 'mine')

  if regions:
    vl = vl_add_region(vl)

  # load the grid, meshes
  # flag region and mine
  idf, meshes = pd_grid_depletion(block_model, regions.split(";"), commalist().parse(mine_include).split(), commalist().parse(mine_exclude).split())
  #idf.mask(idf == -99, inplace=True)
  # calculate the reserves using the grid dataframe
  odf = pd_breakdown(idf, vl)

  if output:
    pd_save_dataframe(odf, output)
  else:
    print(odf.to_string())

  if int(display):
    import matplotlib.pyplot as plt
    vtk_plot_meshes(meshes, False, plt.cm.terrain)

  log("# vtk_reserves finished")


main = vtk_reserves

if __name__=="__main__":
  pyd_zip_extract()
  usage_gui(__doc__)

