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
def normalize(matrix):
    mins = np.percentile(matrix, 0, axis=0)
    maxs = np.percentile(matrix, 100, axis=0) - mins
    
    matrix = (matrix - mins[None,:])/maxs[None,:]
    matrix = np.reshape(matrix, matrix.shape)
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
 sth = shapely.geometry.MultiPolygon([multipoly])
 for poly in sth.geoms:
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

def myIOU(true, pred):
 #1,837,847
 pred_new = tf.argmax(pred, axis=-1)
 pred_new = pred_new[..., tf.newaxis]
 pred_new = tf.dtypes.cast(pred_new, tf.float32)
 metric = 0.0
 intersection = tf.keras.backend.sum(pred_new * true, axis = [-1, -2, -3])
 the_sum = tf.keras.backend.sum(pred_new + true, axis = [-1, -2, -3])
 union = the_sum - intersection
 iou = (intersection + 1e-12)/(union + 1e-12)
 metric = tf.keras.backend.mean(iou)
 return metric

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
 last = tf.keras.layers.Conv2DTranspose(output_ch, 3, strides=1, padding='same', activation='softmax')
 last = last(trconv5)
 model = tf.keras.Model(inputs = inputs, outputs = last)
 print("OK")
 return model
def create_mask(pred_mask):
 print(pred_mask[0][0][0])
 pred_new = tf.argmax(pred_mask, axis=-1)
 print(pred_mask[0][0])
 print("cia")
 pred_new = pred_new[..., tf.newaxis]
 print(pred_mask[0][0])
 return pred_new[0]


zi = zipfile.ZipFile("dstl-satellite-imagery-feature-detection/three_band.zip")


#read polygon data
zitr = zipfile.ZipFile("dstl-satellite-imagery-feature-detection/sample_submission.csv.zip")
fl = zitr.open(zitr.namelist()[0])
train = pd.read_csv(fl)
fl.close()
 
image = {}
polygons = {}
image["6120_2_2"] = readimage("three_band/6120_2_2.tif")
plt.imshow(image["6120_2_2"])
masks = {}
with open('./%s/%s.npy' % ("1", "6120_2_2"), 'rb') as f:
 masks["6120_2_2"] = np.load(f)
poly_pix = {}

W = 3348
H = 3390

W = 3352
H = 3392
train_images_numpy = []
masks_numpy = []
for key in image:
 im = re_shape(image[key], W, H)
 train_images_numpy.append(im[4*W//8:5*W//8, 0*H//8:1*H//8])
 
 ma = re_shape(masks[key], W, H)
 masks_numpy.append(ma[4*W//8:5*W//8, 0*H//8:1*H//8])
del image
del masks
train_images_numpy = np.array(train_images_numpy)
masks_numpy = np.array(masks_numpy)  
  

new_model = tf.keras.models.load_model('saved_modelpadidintast4/my_model', custom_objects={'myIOU':myIOU})
tf.keras.utils.plot_model(new_model, show_shapes=True, dpi=64)
mas = new_model.predict(train_images_numpy)
print(mas.shape)
plt.figure(figsize=(10,10)) 
plt.subplot(1,3, 1)
plt.imshow(train_images_numpy[0])
plt.subplot(1,3, 2)
plt.imshow(tf.keras.preprocessing.image.array_to_img(masks_numpy[0]))
plt.subplot(1,3, 3)
plt.imshow(tf.keras.preprocessing.image.array_to_img(create_mask(mas)))
plt.show()