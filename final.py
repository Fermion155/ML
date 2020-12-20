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
import os
from datetime import datetime
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
 for b in range(0, len(multipoly)):
  for poly in multipoly[b].geoms:
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
      mask[j-1,i-1] = b + 1
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

def getpolygons(build, x, train_images, gri, numberOfObjects):
 polygons = {}
 print(build)
 for i in range(x - 1, len(build["ImageId"])//numberOfObjects*10, 10):
  key = build["ImageId"][i]
  print(key)
  p = shapely.wkt.loads(build["MultipolygonWKT"][i])
  polygons[key] = shapely.geometry.MultiPolygon(p)
 poly_pix = {}
 for key in polygons:
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
  sth = shapely.ops.cascaded_union(tarp)
  if sth.geom_type != 'MultiPolygon':
   poly_pix[key] = shapely.geometry.MultiPolygon([sth])
  else:
   poly_pix[key] = sth
 return poly_pix

def myIOU(true, pred):
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

def printTime():
 now = datetime.now()

 current_time = now.strftime("%H:%M:%S")
 print("Current Time =", current_time)

zi = zipfile.ZipFile("dstl-satellite-imagery-feature-detection/three_band.zip")

printTime()
#read polygon data

zitr = zipfile.ZipFile("dstl-satellite-imagery-feature-detection/train_wkt_v4.csv.zip")
fl = zitr.open(zitr.namelist()[0])
train = pd.read_csv(fl)
fl.close()
print(train["MultipolygonWKT"][0])

#build = train.loc[(train["ClassType"] == 1) | (train["ClassType"] == 3) | (train["ClassType"] == 9) | (train["ClassType"] == 10)]
build = train.loc[train["ClassType"] == 1]  #get only buildings
numberOfObjects = 1
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
for i in range(0, len(build["ImageId"])//numberOfObjects*10, 10):
 key = build["ImageId"][i]
 print(key)
 train_images[key] = readimage("three_band/" + key + ".tif")
poly_pix = {}
indexes = ["1", "3", "9", "10"]
masks = {}
dirname = ''
for i in range(0, numberOfObjects):
 dirname = dirname + indexes[i]


if not os.path.exists('./%s' % dirname):
 os.mkdir('./%s' % dirname)
 for i in range(0, numberOfObjects):
  poly_pix[indexes[i]] = getpolygons(build, int(indexes[i]), train_images, gri, numberOfObjects)  

 print(poly_pix["1"])

 for key in train_images:
  masforpix = []
  for i in range(0, numberOfObjects):
   masforpix.append(poly_pix[indexes[i]][key])
  print("creating mask:")
  print(key)
  #masks[key] = np.array([pix_to_masks(train_images[key], [poly_pix["1"][key], poly_pix["3"][key], poly_pix["9"][key], poly_pix["10"][key]])]).transpose([1,2,0])
  masks[key] = np.array([pix_to_masks(train_images[key], masforpix)]).transpose([1,2,0])
  with open('./%s/%s.npy' % (dirname, key), 'wb') as f:
   np.save(f, masks[key])
else:
 for key in train_images:
  with open('./%s/%s.npy' % (dirname, key), 'rb') as f:
   masks[key] = np.load(f)


W = 3352
H = 3392
masks_numpy = []

train_images_numpy = []
parts = 8
for key in train_images:
 im = re_shape(train_images[key], W, H)
 for i in range(0, parts):
  for j in range(0, parts):
   train_images_numpy.append(im[j*W//parts:(j+1)*W//parts, i*H//parts:(i+1)*H//parts])
 

 ma = re_shape(masks[key], W, H)
 for i in range(0, parts):
  for j in range(0, parts):
   masks_numpy.append(ma[j*W//parts:(j+1)*W//parts, i*H//parts:(i+1)*H//parts])
 
 print("masks: ")
 print(masks[key].shape)
del masks
del train_images 
train_images_numpy = np.array(train_images_numpy)
masks_numpy = np.array(masks_numpy)  
  


#categorical
#masks_numpy = tf.keras.utils.to_categorical(masks_numpy, num_classes=2)
print(masks_numpy.shape)
print(train_images_numpy.shape)


#flipping images (data augmention)
arr2 = np.arange(masks_numpy.shape[0])
np.random.shuffle(arr2)
printTime()
arrayilgis = masks_numpy.shape[0]
tarpmask = []
tarpimag = []
for i in range(0, 400):
 print(i)
 tarpmask.append(masks_numpy[arr2[i],::-1,...])
 print(str(i) + "middle")
 tarpimag.append(train_images_numpy[arr2[i],::-1,...])
 

masks_numpy = np.append(masks_numpy, tarpmask, axis = 0)
del tarpmask
print("done")
train_images_numpy = np.append(train_images_numpy, tarpimag, axis = 0)
del tarpimag
print("superdone")
masks_numpy = np.array(masks_numpy)
train_images_numpy = np.array(train_images_numpy)
printTime()


#train test sets
arr = np.arange(masks_numpy.shape[0])
print(arr)
np.random.shuffle(arr)
test_images = []
test_masks = []
train_images = []
train_masks = []
spliting = int(len(arr) * 0.2)
for i in range(0, spliting):
 test_images.append(train_images_numpy[arr[i]])
 test_masks.append(masks_numpy[arr[i]])
for i in range(spliting, len(arr)):
 train_images.append(train_images_numpy[arr[i]])
 train_masks.append(masks_numpy[arr[i]])
del masks_numpy
del train_images_numpy
test_images = np.array(test_images)
test_masks = np.array(test_masks)
train_images = np.array(train_images)
train_masks = np.array(train_masks)


print(test_images.shape)
print(test_masks.shape)
print(train_images.shape)
print(train_masks.shape)

callback = tf.keras.callbacks.EarlyStopping(monitor='val_myIOU', patience=10, mode="max", restore_best_weights=True)

model = create_unet_model(W//parts, H//parts, 3, 2)
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=[myIOU, 'accuracy'])

#tf.keras.metrics.MeanIoU(num_classes=2)
EPOCHS = 60
printTime()
BATCH_SIZE = 4
history = model.fit(train_images, train_masks, BATCH_SIZE, epochs=EPOCHS, validation_split = 0.2, callbacks = [callback])

model.save('saved_modelpadidintas4/my_model')
printTime()
losse, ioue, acce = model.evaluate(test_images, test_masks, BATCH_SIZE)
print("Loss: ", losse)
print("Iou: ", ioue)
print("Acc: ", acce) 
printTime()
#check history
history_dict = history.history
print(history_dict.keys())

acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss'] 
iou = history_dict['myIOU']
val_iou = history_dict['val_myIOU']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

plt.show()

plt.plot(epochs, iou, 'bo', label='Training iou')
plt.plot(epochs, val_iou, 'b', label='Validation iou')
plt.title('Training and validation iou')
plt.xlabel('Epochs')
plt.ylabel('Iou')
plt.legend(loc='lower right')

plt.show()
