import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff
import zipfile
import matplotlib
#from rasterio.plot import reshape_as_image, adjust_band, show
import pandas as pd
import shapely.wkt
import shapely.geometry
import shapely.ops

def normalize(arr):
 arr_min = 255
 arr_max = 0
 """for i in range(0, len(arr)):
  for j in range(0, len(arr[i])):
   if arr_min > arr[i][j].min():
    arr_min = arr[i][j].min()
   if arr_max < arr[i][j].max()
   arr_max = arr[i][j].max()"""
 arr_min = arr.min()
 arr_max = arr.max()
 return ((arr - arr_min) / (arr_max - arr_min)).astype(float)
def scale_percentile(matrix):
    """Fixes the pixel value range to 2%-98% original distribution of values"""
    orig_shape = matrix.shape
    matrix = np.reshape(matrix, [matrix.shape[0]*matrix.shape[1], 3]).astype(float)
    
    # Get 2nd and 98th percentile
    mins = np.percentile(matrix, 2, axis=0)
    maxs = np.percentile(matrix, 98, axis=0) - mins
    
    matrix = (matrix - mins[None,:])/maxs[None,:]
    matrix = np.reshape(matrix, orig_shape)
    matrix = matrix.clip(0,1)
    return matrix
def funk(zi):
 for name in zi.namelist():
  #print(name)
  if name == "three_band/6120_2_2.tif":
   fl = zi.open(name)
   P = tiff.imread(fl)
   fl.close()
   #print(P)
   return P  
zi = zipfile.ZipFile("dstl-satellite-imagery-feature-detection/three_band.zip")
#print(zi.namelist())
#P = tiff.imread("dstl-satellite-imagery-feature-detection/6010_0_0.tif")
#P[0,:,:] = normalize(P[0,:,:])
#P[1,:,:] = normalize(P[1,:,:])
#P[2,:,:] = normalize(P[2,:,:])
#P = normalize(P)
#P = np.rollaxis(P, 0, 3)
#P = P.transpose([1, 2, 0])
#rgb_norm = adjust_band(P) # normalize bands to range between 1.0 to 0.0
#rgb_reshaped = reshape_as_image(rgb_norm) # reshape to [rows, cols, bands]
#print(P.shape)
#fixed_im = scale_percentile(P)
#fixed_im = normalize(P)
#P = P.astype(float)
#P[:,:,0] = normalize(P[:,:,0])
#P[:,:,1] = normalize(P[:,:,1])
#P[:,:,2] = normalize(P[:,:,2])
#tiff.imshow(P, figure = plt,interpolation='nearest')
#show(rgb_norm, ax=plt)
#plt.imshow(fixed_im)

plt.figure(figsize=(10,10))
for i in range(0, 0):
 plt.subplot(5,5,i+1)
 plt.xticks([])
 plt.yticks([])
 plt.grid(False)
 fl = zi.open(zi.namelist()[i])
 P = tiff.imread(fl)
 fl.close() 
 #print(P.shape)
 #print(P[:,300,200])
 P = P.transpose([1,2,0])
 nor = scale_percentile(P)
 #P = P/255.0
 #print(P[:,300,200])
 print(i)
 #tiff.imshow(nor, figure = plt, subplot=((55*10)+i + 1))
 plt.imshow(nor)
 #plt.imshow(P[:,:,0], cmap=plt.cm.hsv)
 #plt.imshow(P[:,:,1], cmap=plt.cm.green)
 #plt.imshow(P[:,:,2], cmap=plt.cm.blue)
 #plt.imshow(train_images[i], cmap=plt.cm.binary)
 #plt.xlabel(class_names[train_labels[i]])



plt.show()
zitr = zipfile.ZipFile("dstl-satellite-imagery-feature-detection/train_wkt_v4.csv.zip")
fl = zitr.open(zitr.namelist()[0])
train = pd.read_csv(fl)
fl.close()
print(train["MultipolygonWKT"][0])
build = train.loc[train["ClassType"] == 1]
zigr = zipfile.ZipFile("dstl-satellite-imagery-feature-detection/grid_sizes.csv.zip")
fl = zigr.open(zigr.namelist()[0])
gri = pd.read_csv(fl)
fl.close()

P = funk(zi)
#print(P)
P = P.transpose([1,2,0])
W = P.shape[1]
H = P.shape[0]
Ws = W * W/(W+1)
Hs = H * H/(H+1)
print(gri["Unnamed: 0"])
xmax = 0
ymin = 0
for i in range(0, len(gri["Unnamed: 0"])):
 if gri["Unnamed: 0"][i] == "6120_2_2":
  print("rado")
  xmax = gri["Xmax"][i]
  ymin = gri["Ymin"][i]
  
xn = Ws/xmax
yn = Hs/ymin
nor = scale_percentile(P)
plt.imshow(nor)
#plt.show()
print(P.shape)
print(build)
poly = [shapely.wkt.loads(p) for p in build["MultipolygonWKT"]]
#print(poly[1])
mult = shapely.geometry.MultiPolygon(poly[1])
#x,y = mult.exterior.xy
#plt.plot(x,y)
print(gri)
lit = []
for geom in mult.geoms:
 lit.append(shapely.affinity.scale(geom, xfact = xn, yfact = yn, origin = (0,0)))
mch = shapely.ops.cascaded_union(lit)

for geom in mch.geoms:    
 xs, ys = geom.exterior.xy    
 #plt.fill(xs, ys, alpha=0.5, fc='r', ec='none')
 #plt.imshow([xs,ys])
 plt.fill(xs, ys, facecolor='none', edgecolor='purple', linewidth=1)
plt.show()

siz = nor.shape[:2]
print(siz)
mask = np.zeros(siz)
l = 0
for poly in mch.geoms:
 print(l)
 minx = int(poly.bounds[0])
 miny = int(poly.bounds[1])
 maxx = int(poly.bounds[2])
 maxy = int(poly.bounds[3])
 l = l + 1
 if minx <= 0:
  minx = 1
 if miny <= 0:
  miny = 1
 if maxx >= siz[1]:
  maxx = siz[1] + 1
 if maxy >= siz[0]:
  maxy = siz[0] + 1
 for i in range(minx, maxx):
  for j in range(miny, maxy):
   point = shapely.geometry.Point(i,j)
   if point.intersects(poly):
    mask[j-1,i-1] = 1

plt.imshow(mask)
plt.show()

"""ttt = shapely.geometry.Polygon([(-1, 1), (2, 1), (2, 2)])
ttp = shapely.geometry.Polygon([(3, 1), (4, 1), (4, 2), (3, 2)])
ttd = shapely.affinity.scale(ttt, xfact = 2.0, yfact = 3.2, origin = (0,0))
ttpd = shapely.affinity.scale(ttp, xfact = 2.0, yfact = 3.2, origin = (0,0))
xs, ys = ttd.exterior.xy
plt.fill(xs,ys, facecolor = 'red', edgecolor='red')
xs, ys = ttpd.exterior.xy
plt.fill(xs,ys, facecolor = 'blue', edgecolor='blue')
plt.show()
li = []
mmm = shapely.geometry.MultiPolygon([ttt,ttp])
for geom in mmm.geoms:
 li.append(shapely.affinity.scale(geom, xfact = 2.0, yfact = 3.2, origin = (0,0)))
mch = shapely.ops.cascaded_union(li)
for geom in mch.geoms:
 xs, ys = geom.exterior.xy
 plt.fill(xs,ys, facecolor = 'blue', edgecolor='blue')
plt.show()"""