#!python
# save a triangulation in vulcan format
# binary or ascii
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

https://github.com/pemn/usage-gui
---------------------------------

'''
import numpy as np
import pandas as pd
import os.path
import skimage.io
import re
#from scipy import ndimage as ndi
#from affine import Affine

def vulcan_save_asc(nodes, faces, output):
  of = open(output, 'w')
  for i in range(3):
    print("%-24s" % "Created External", file=of)

  print("No Points: %d, No Triangles: %d" % (len(nodes), len(faces)), file=of)
  for n in nodes:
    print("Vertex: %16.4f, %16.4f, %16.4f" % tuple(n), file=of)
  for f in faces:
    print("Index : %06d, %06d, %06d" % tuple(np.add(f,(1,1,1))), file=of)

def vulcan_save_tri(nodes, faces, output, colour = 1):
  try:
    import vulcan
  except:
    print("vulcan module not found. failed to save file:", output)
    return
  tri = vulcan.triangulation("", "w")
  tri.set_colour(colour)
  for k in nodes:
    tri.add_node(*k)
  for k in faces:
    tri.add_face(*map(int, k))
  tri.save(output)


# Vulcan Triangulation 00t
def vulcan_load_tri(df_path):
  import vulcan
  import pandas as pd
  tri = vulcan.triangulation(df_path)
  cv = tri.get_colour()
  cn = 'colour'
  if vulcan.version_major >= 11 and tri.is_rgb():
    cv = pd.np.sum(pd.np.multiply(tri.get_rgb(), [2**16,2**8,1]))
    cn = 'rgb'

  nodes = [tri.get_node(_) for _ in range(tri.n_nodes())]
  faces = [tri.get_face(_) for _ in range(tri.n_faces())]

  return nodes, faces, cv, cn

# por each vertex, create a texture mapping node
def vulcan_texture_vt(cols, rows):
  x_grid, y_grid = np.meshgrid(np.linspace(0, 1, rows), np.linspace(1, 0, cols))
  return np.column_stack((x_grid.flat, y_grid.flat))

# save a triangulation as a Wavefront OBJ (obj, mtl, png)
def vulcan_save_obj(nodes, faces, texture, output_path, rows_cols = None):
  # obj file
  of = open(output_path, 'w')
  output_mtl = os.path.splitext(output_path)[0] + '.mtl'

  print("mtllib", output_mtl, file=of)
  print("usemtl material0", file=of)
  for n in nodes:
    print("v %f %f %f" % tuple(n), file=of)

  if rows_cols is not None:
    for uv in vulcan_texture_vt(*rows_cols):
      print("vt %f %f" % tuple(uv.tolist()), file=of)

  for f in faces:
    face1 = np.add(f,(1,1,1))
    print("f %d/%d %d/%d %d/%d" % tuple(np.column_stack((face1, face1)).flat), file=of)

  of.close()
  # tif file
  output_img = os.path.splitext(output_path)[0] + '.png'
  skimage.io.imsave(output_img, texture)

  # mtl file
  of = open(output_mtl, 'w')
  print("newmtl material0", file=of)
  print("Ka %f %f %f" % (1.0, 1.0, 1.0), file=of)
  print("Kd %f %f %f" % (1.0, 1.0, 1.0), file=of)
  print("Ks %f %f %f" % (0.0, 0.0, 0.0), file=of)
  print("map_Kd", output_img, file=of)
  of.close()

def get_boilerplate_json(output_img, output_00t):
  return {
    "properties": 
    {
     "bounding_level": 0.0,
     "highlight_col": 65535,
     "image": output_img,
     "image_col": 16777215,
     "scale": 1000.0,
     "sharp_pixels": 1,
     "triangulation": output_00t,
     "tricol": 0,
     "undercol": 16777215,
     "use_bounding": 1,
     "use_specified": 0,
     "world_col": 16777215
    }
  }

def vulcan_register_image(output_00t, texture, xyz, output_path):
  import json
  output_img = os.path.splitext(output_path)[0] + '.png'

  spec_json = get_boilerplate_json(output_img, output_00t)
  skimage.io.imsave(output_img, texture)
  spec_json["points"] = []
  spec_json["points"].append({"image": [0,0,0],"world": xyz[0]})
  spec_json["points"].append({"image": [1,0,0],"world": xyz[1]})
  spec_json["points"].append({"image": [1,1,0],"world": xyz[2]})
  spec_json["points"].append({"image": [0,1,0],"world": xyz[3]})

  open(output_path, 'w').write(json.dumps(spec_json, sort_keys=True, indent=4).replace(': NaN', ' = u').replace('": ', '" = '))

# save a triangulation as a Vulcan IREG (ireg, 00t, png)
def vulcan_save_ireg(nodes, faces, texture, output_path, rows_cols = None):
  import json
  spec_json = get_boilerplate_json(output_img, output_00t)

  output_00t = os.path.splitext(output_path)[0] + '.00t'
  vulcan_save_tri(nodes, faces, output_00t)
  
  output_img = os.path.splitext(output_path)[0] + '.png'
  skimage.io.imsave(output_img, texture)

  if rows_cols is not None:
    vt = vulcan_texture_vt(*rows_cols)

    spec_json["points"] = [{"image": vt[i].tolist(),"world": nodes[i].tolist()} for i in range(len(vt))]
 

  open(output_path, 'w').write(json.dumps(spec_json, sort_keys=True, indent=4).replace(': NaN', ' = u').replace('": ', '" = '))

# 29193
def gdal_save_geotiff(texture, xyz, output_path, epsg = 29193):
  import osgeo.gdal as gdal
  import osgeo.osr as osr
  
  driver = gdal.GetDriverByName("GTiff")
  gdt = gdal.GDT_Byte
  if str(texture.dtype).startswith('float'):
    gdt = gdal.GDT_Float32
  ds = driver.Create(output_path, texture.shape[2], texture.shape[1], texture.shape[0], gdt, options = ['PHOTOMETRIC=RGB'])
  if xyz is not None:
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    gcps = []
    gcps.append(gdal.GCP(xyz[0][0],xyz[0][1],xyz[0][2],0,0))
    gcps.append(gdal.GCP(xyz[1][0],xyz[1][1],xyz[1][2],1,0))
    gcps.append(gdal.GCP(xyz[2][0],xyz[2][1],xyz[2][2],1,1))
    gcps.append(gdal.GCP(xyz[3][0],xyz[3][1],xyz[3][2],0,1))
    ds.SetGCPs(gcps, srs.ExportToWkt())

  for i in range(texture.shape[0]):
    ds.GetRasterBand(i+1).WriteArray(texture[i])
  ds.FlushCache()

def pd_load_geotiff(input_path):
  ''' create a standadized dataframe from a geotiff '''
  import osgeo.gdal as gdal
  import osgeo.osr as osr
  import skimage.transform
  
  idset = gdal.Open(input_path)
  sr = idset.GetSpatialRef()
  an = 'Authority'
  ac = None
  if sr:
    print(sr.GetName())
    an = sr.GetAuthorityName(None)
    if not an:
      an = 'EPSG'
    ac = sr.GetAuthorityCode(None)
    if ac:
      ac = int(ac)

  gt = idset.GetGeoTransform()
  # ReadAsArray(self, xoff=0, yoff=0, xsize=None, ysize=None, buf_obj=None, buf_xsize=None, buf_ysize=None, buf_type=None, resample_alg=gdalconst.GRIORA_NearestNeighbour, callback=None, callback_data=None, interleave='band')
  bd = idset.ReadAsArray(0, 0, idset.RasterXSize, idset.RasterYSize)
  #bd = bd.astype(np.float)

  vl = []
  for i in range(idset.RasterCount):
    vl.append(str(i))
    # sometime gdal does not handle NoData corretly so we do it again anyway
    nodata = idset.GetRasterBand(i+1).GetNoDataValue()
    if nodata is None:
      # nodata not defined
      pass
    elif str(bd.dtype).find('int') >= 0:
      # int types do not support NaN
      pass
    elif bd.ndim == 2:
      np.putmask(bd, bd == nodata, np.nan)
    else:
      np.putmask(bd[i], bd[i] == nodata, np.nan)

  # channels
  oa = None
  n2d = np.prod(bd.shape[bd.ndim - 2:])
  # single channel
  if bd.ndim == 2:
    oa = np.reshape(bd.T, (n2d, 1))
  else:
    oa = np.transpose(bd, (2,1,0)).reshape((n2d, bd.shape[0]))
  affmat = np.array([[gt[1], gt[2], gt[0]], [gt[4], gt[5], gt[3]], [0.0,0.0,1.0]])
  # xc, yc
  vbi = np.indices((idset.RasterXSize, idset.RasterYSize))
  vbi = np.moveaxis(vbi, 0, -1)
  vbi = vbi.reshape(n2d, 2)
  # x,y
  vbp = skimage.transform.matrix_transform(vbi, affmat)
  # epsg,x0,txx,txxy,y0,tyx,tyy
  vbr = np.tile((ac,) + gt, (n2d,1))
  
  vbv = np.concatenate((vbr, vbi, vbp, oa), 1)

  return pd.DataFrame(vbv, columns=[an, 'x0', 'txx', 'txy', 'y0', 'tyx', 'tyy', 'xc', 'yc', 'x', 'y'] + vl)

def pd_save_geotiff(df, output_path):
  import osgeo.gdal as gdal
  import osgeo.osr as osr
  driver = gdal.GetDriverByName("GTiff")
  nx = int(df['xc'].max()) + 1
  ny = int(df['yc'].max()) + 1
  nc = sum(map(str.isnumeric, df.columns))
  gdt = gdal.GDT_Byte
  if re.search('float|object', str(df['0'].dtype)):
    gdt = gdal.GDT_Float32
  if nc <= 1:
    nc = 1
    ds = driver.Create(output_path, nx, ny, nc, gdt, options = ['PROFILE=GeoTIFF'])
  else:
    ds = driver.Create(output_path, nx, ny, nc, gdt, options = ['PROFILE=GeoTIFF', 'PHOTOMETRIC=RGB'])
  an = df.columns[0]
  ac = df.iloc[0, 0]
  print("ac",ac)
  if ac:
    sr = osr.SpatialReference()
    sr.SetFromUserInput("%s:%d" % (an, ac))
    ds.SetSpatialRef(sr)
    print(sr.GetName())
  dscol = ['x0', 'txx', 'txy', 'y0', 'tyx', 'tyy']
  # check if all required columns are in df
  if set(dscol).issubset(df.columns):
    ds.SetGeoTransform(df.iloc[0][dscol])
  for i in range(ds.RasterCount):
    ds.GetRasterBand(i+1).WriteArray(df[str(i)].values.reshape((nx,ny)).T)
  ds.FlushCache()


def obj_mesh_to_ireg(od, output_img, output_path):
  import json
  output_00t = os.path.splitext(output_path)[0] + '.00t'
  vulcan_save_tri(od.get('v'), od.get('f'), output_00t)
  spec_json = get_boilerplate_json(output_img, output_00t)
  nodes = od.get('v')
  vt = od.get('vt')
  spec_json["points"] = [{"image": vt[i],"world": nodes[i]} for i in range(len(vt))]

  open(output_path, 'w').write(json.dumps(spec_json, sort_keys=True, indent=4).replace(': NaN', ' = u').replace('": ', '" = '))
