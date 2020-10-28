import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff
import zipfile
import matplotlib
import pandas as pd
import shapely.wkt
import shapely.geometry
import shapely.ops
import pydot
import pydotplus
from pydotplus import graphviz


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
def readimage(name):
 print(name)
 fl = zi.open(name)
 P = tiff.imread(fl)
 fl.close()
 P = P.transpose([1,2,0])
 nor = scale_percentile(P)
 return nor  
def pix_to_masks(image, multipoly):
 siz = image.shape[:2]
 mask = np.zeros(siz)
 for poly in multipoly.geoms:
  minx = int(poly.bounds[0])
  miny = int(poly.bounds[1])
  maxx = int(poly.bounds[2])
  maxy = int(poly.bounds[3])
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
 return mask
def re_shape(image, W, H):
 if image.shape[0] > W:
  image = image[:W]
 if image.shape[1] > H:
  k = image.shape[0]
  print(image.shape[1])
  image = image[0:k, 0:H]
  print(image.shape[1])
  print("pabaiga")
 if image.shape[0] < W:
  for i in range(image.shape[0], W):
   temp = np.array([])
   #image = np.concatenate((image, []), axis = 0)
   for j in range(0, image.shape[1]):
    if j == 0:
     if len(image.shape)> 2:
      temp = np.array([[[0]*image.shape[2]]])
     else:
      temp = np.array([[0]*image.shape[1]])
      break
    else:     
     temp = np.concatenate((temp, np.array([[[0] * image.shape[2]]])), axis = 1)
   image = np.concatenate((image, temp), axis = 0)   
   
   
 if image.shape[1] < H:
  for i in range(image.shape[1], H):
   temp = np.array([])
   for j in range(0, image.shape[0]):
    if j == 0:
     if len(image.shape)> 2:
      temp = np.array([[[0]* image.shape[2]]])
     else:
      temp = np.array([[0]])
    else:
     if len(image.shape)> 2:
      temp = np.concatenate((temp, np.array([[[0] * image.shape[2]]])), axis = 0)
     else:
      temp = np.concatenate((temp, np.array([[0]])), axis = 0)
   image = np.concatenate((image, temp), axis = 1)
 return image

def create_unet_model(W, H, input_ch ,output_ch):
 inputs = tf.keras.layers.Input(shape = (W, H, input_ch))
 #downsampling
 conv1 = tf.keras.layers.Conv2D(16, 3, strides = 1, padding = "same", activation='relu')
 conv1 = conv1(inputs)
 pool1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)
 pool1 = pool1(conv1)
 conv2 = tf.keras.layers.Conv2D(32, 3, strides = 1, padding = "same", activation='relu')
 conv2 = conv2(pool1)
 pool2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)
 pool2 = pool2(conv2)
 conv3 = tf.keras.layers.Conv2D(64, 3, strides = 1, padding = "same", activation='relu')
 conv3 = conv3(pool2)
 pool3 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)
 pool3 = pool3(conv3)
 conv4 = tf.keras.layers.Conv2D(128, 3, strides = 1, padding = "same", activation='relu')
 conv4 = conv4(pool3)
 pool4 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)
 pool4 = pool4(conv4)
 conv5 = tf.keras.layers.Conv2D(256, 3, strides = 1, padding = "same", activation='relu')
 conv5 = conv5(pool4)
 pool5 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)
 pool5 = pool5(conv5) 
 conv6 = tf.keras.layers.Conv2D(256, 3, strides = 1, padding = "same", activation='relu')
 conv6 = conv6(pool5)
 #upsampling
 trconv1 = tf.keras.layers.Conv2DTranspose(256, 3, strides = 2, padding='valid', activation='relu')
 trconv1 = trconv1(conv6)
 reshape = tf.keras.layers.experimental.preprocessing.Resizing(conv5.shape[1], conv5.shape[2])
 trconv1 = reshape(trconv1)
 trconv1 = tf.keras.layers.Concatenate()([conv5, trconv1])
 trconv2 = tf.keras.layers.Conv2DTranspose(128, 3, strides = 2, padding='valid', activation='relu')
 trconv2 = trconv2(trconv1)
 #print(conv4.shape)
 reshape = tf.keras.layers.experimental.preprocessing.Resizing(conv4.shape[1], conv4.shape[2])
 trconv2 = reshape(trconv2)
 trconv2 = tf.keras.layers.Concatenate()([conv4, trconv2])
 trconv3 = tf.keras.layers.Conv2DTranspose(64, 3, strides = 2, padding='valid', activation='relu')
 trconv3 = trconv3(trconv2)
 reshape = tf.keras.layers.experimental.preprocessing.Resizing(conv3.shape[1], conv3.shape[2])
 trconv3 = reshape(trconv3)
 trconv3 = tf.keras.layers.Concatenate()([conv3, trconv3])
 
 trconv4 = tf.keras.layers.Conv2DTranspose(32, 3, strides = 2, padding='valid', activation='relu')
 trconv4 = trconv4(trconv3)
 reshape = tf.keras.layers.experimental.preprocessing.Resizing(conv2.shape[1], conv2.shape[2])
 trconv4 = reshape(trconv4)
 trconv4 = tf.keras.layers.Concatenate()([conv2, trconv4])
 
 trconv5 = tf.keras.layers.Conv2DTranspose(16, 3, strides = 2, padding='valid', activation='relu')
 trconv5 = trconv5(trconv4)
 reshape = tf.keras.layers.experimental.preprocessing.Resizing(conv1.shape[1], conv1.shape[2])
 trconv5 = reshape(trconv5)
 trconv5 = tf.keras.layers.Concatenate()([conv1, trconv5]) 
 
   # This is the last layer of the model
 last = tf.keras.layers.Conv2DTranspose(output_ch, 3, strides=1, padding='same')
 last = last(trconv5)
 model = tf.keras.Model(inputs = inputs, outputs = last)
 tf.keras.utils.plot_model(model, show_shapes=True, dpi=64)
 print("OK")
 return model



zi = zipfile.ZipFile("dstl-satellite-imagery-feature-detection/three_band.zip")


#plt.show()
#read polygon data
zitr = zipfile.ZipFile("dstl-satellite-imagery-feature-detection/train_wkt_v4.csv.zip")
fl = zitr.open(zitr.namelist()[0])
train = pd.read_csv(fl)
fl.close()
print(train["MultipolygonWKT"][0])

build = train.loc[train["ClassType"] == 1]  #get only buildings
#read grid sizes
zigr = zipfile.ZipFile("dstl-satellite-imagery-feature-detection/grid_sizes.csv.zip")

fl = zigr.open(zigr.namelist()[0])
gri = pd.read_csv(fl)
fl.close()
#read images
train_images = {}
l = 0
polygons = {}
print(build)
for i in range(0, len(build["ImageId"])*10, 10):
 key = build["ImageId"][i]
 #print("A")
 p = shapely.wkt.loads(build["MultipolygonWKT"][i])
 polygons[key] = shapely.geometry.MultiPolygon(p)
 #print("cia")
 train_images[key] = readimage("three_band/" + key + ".tif")
 #print(i)
poly_pix = {}
for key in polygons:
 #print(key)
 W = train_images[key].shape[1]
 H = train_images[key].shape[0]
 Ws = W * W/(W+1)
 Hs = H * H/(H+1)
 xmax = 0
 ymin = 0
 for i in range(0, len(gri["Unnamed: 0"])):
  if gri["Unnamed: 0"][i] == key:
   xmax = gri["Xmax"][i]
   ymin = gri["Ymin"][i]
   break
 xn = Ws/xmax
 yn = Hs/ymin
 tarp = []
 for geom in polygons[key].geoms:
  tarp.append(shapely.affinity.scale(geom, xfact = xn, yfact = yn, origin = (0,0)))
 poly_pix[key] = shapely.ops.cascaded_union(tarp)
  
masks = {}
for key in train_images:
 #print(train_images[key].shape)
 masks[key] = np.array([pix_to_masks(train_images[key], poly_pix[key])]).transpose([1,2,0])
W = 3348
#H = 3391
H = 3388
masks_numpy = []

train_images_numpy = []

for key in train_images:
 im = re_shape(train_images[key], W, H)
 train_images_numpy.append(im[0:W//4, 0:H//4])
 train_images_numpy.append(im[W//4:2*W//4, 0:H//4])
 train_images_numpy.append(im[2*W//4:3*W//4, 0:H//4])
 train_images_numpy.append(im[3*W//4:W, 0:H//4])
 
 train_images_numpy.append(im[0:W//4, H//4:2*H//4])
 train_images_numpy.append(im[W//4:2*W//4, H//4:2*H//4])
 train_images_numpy.append(im[2*W//4:3*W//4, H//4:2*H//4])
 train_images_numpy.append(im[3*W//4:W, H//4:2*H//4])

 train_images_numpy.append(im[0:W//4, 2*H//4:3*H//4])
 train_images_numpy.append(im[W//4:2*W//4, 2*H//4:3*H//4])
 train_images_numpy.append(im[2*W//4:3*W//4, 2*H//4:3*H//4])
 train_images_numpy.append(im[3*W//4:W, 2*H//4:3*H//4]) 
 
 train_images_numpy.append(im[0:W//4, 3*H//4:H])
 train_images_numpy.append(im[W//4:2*W//4, 3*H//4:H])
 train_images_numpy.append(im[2*W//4:3*W//4, 3*H//4:H])
 train_images_numpy.append(im[3*W//4:W, 3*H//4:H]) 
 

 ma = re_shape(masks[key], W, H)
 
 masks_numpy.append(ma[0:W//4, 0:H//4])
 masks_numpy.append(ma[W//4:2*W//4, 0:H//4])
 masks_numpy.append(ma[2*W//4:3*W//4, 0:H//4])
 masks_numpy.append(ma[3*W//4:W, 0:H//4])
 
 masks_numpy.append(ma[0:W//4, H//4:2*H//4])
 masks_numpy.append(ma[W//4:2*W//4, H//4:2*H//4])
 masks_numpy.append(ma[2*W//4:3*W//4, H//4:2*H//4])
 masks_numpy.append(ma[3*W//4:W, H//4:2*H//4])

 masks_numpy.append(ma[0:W//4, 2*H//4:3*H//4])
 masks_numpy.append(ma[W//4:2*W//4, 2*H//4:3*H//4])
 masks_numpy.append(ma[2*W//4:3*W//4, 2*H//4:3*H//4])
 masks_numpy.append(ma[3*W//4:W, 2*H//4:3*H//4]) 
 
 masks_numpy.append(ma[0:W//4, 3*H//4:H])
 masks_numpy.append(ma[W//4:2*W//4, 3*H//4:H])
 masks_numpy.append(ma[2*W//4:3*W//4, 3*H//4:H])
 masks_numpy.append(ma[3*W//4:W, 3*H//4:H]) 
 
 print("masks: ")
 print(masks[key].shape)
del masks
del train_images 
train_images_numpy = np.array(train_images_numpy)
masks_numpy = np.array(masks_numpy)  
  
#plt.figure(figsize=(10,10))  

#for key in train_images:
 #print(train_images[key].shape)
 #print(masks[key].shape)
 #plt.subplot(1,2, 1)
 #plt.imshow(train_images[key])
 #plt.subplot(1,2, 2)
 #for geom in poly_pix[key].geoms:
  #xs, ys = geom.exterior.xy    
  #plt.fill(xs, ys, facecolor='none', edgecolor='purple', linewidth=1)
 #plt.imshow(tf.keras.preprocessing.image.array_to_img(masks[key]))
 #plt.show()

plt.figure(figsize=(10,10)) 
plt.subplot(2,2, 1)
plt.imshow(train_images_numpy[16])
plt.subplot(2,2, 2)
plt.imshow(tf.keras.preprocessing.image.array_to_img(masks_numpy[16]))
plt.subplot(2,2, 3)
plt.imshow(train_images_numpy[17])
plt.subplot(2,2, 4)
plt.imshow(tf.keras.preprocessing.image.array_to_img(masks_numpy[17]))
plt.show()
print(masks_numpy.shape)
print(train_images_numpy.shape)
model = create_unet_model(W//4, H//4, 3, 2)
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


EPOCHS = 10

model.fit(train_images_numpy, masks_numpy, 1, epochs=EPOCHS)

model.save('saved_model/my_model')
