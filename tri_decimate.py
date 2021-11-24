#!python
# simplifies a triangulation using vtk decimate
# target_reduction: proportion of faces to be removed
# display: show input and output side by side in 3d
# v1.0 09/2020 paulo.ernesto
# Copyright 2020 Vale
# License: Apache 2.0
"""
usage: $0 input_path*00t,obj,msh,vtk,vtp,ply,stl target_reduction=0.5 output*00t,msh,vtk,vtp,ply,stl display@
"""

import sys, os.path

# import modules from a pyz (zip) file with same name as scripts
sys.path.insert(0, os.path.splitext(sys.argv[0])[0] + '.pyz')

from _gui import pyd_zip_extract, usage_gui, pd_load_dataframe, pd_save_dataframe, df_to_nodes_faces

from vulcan_save_tri import vulcan_save_tri
from tri_leapfrog_convert import leapfrog_save_mesh, leapfrog_load_mesh
import numpy as np
import pandas as pd

pyd_zip_extract()

import pyvista as pv
#from vulcanvtk import tri_to_vtk
from pd_vtk import vtk_nf_to_mesh, pv_read, pv_save

# convert a vulcan surface to a vulcan solid
def main(input_path, target_reduction, output, display):
  if target_reduction:
    target_reduction = float(target_reduction)
  else:
    target_reduction = 0.0
  # australian english...
  colour = 1
  imesh = pv_read(input_path)

  if target_reduction == 0:
    omesh = imesh
    print("n_faces unchanged", imesh.n_faces)
  else:
    print("n_faces input", imesh.n_faces)
    # omesh = imesh.decimate(target_reduction)
    omesh = imesh.decimate_pro(target_reduction)
    print("n_faces output", omesh.n_faces)

  if output:
    pv_save([omesh], output)

  if int(display):
    p = pv.Plotter(shape=(1, 2))
    p.subplot(0, 0)
    p.add_mesh(imesh)
    p.add_text("input n_faces %d" % imesh.n_faces)
    p.subplot(0, 1)
    p.add_mesh(omesh)
    p.add_text("output n_faces %d" % omesh.n_faces)
    p.show()

if __name__=="__main__":
  usage_gui(__doc__)
