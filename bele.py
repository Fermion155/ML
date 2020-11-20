
import os
import numpy as np

indexes = ["1", "3", "9", "10"]
numberOfObjects = 1
dirname = ''
for i in range(0, numberOfObjects):
 dirname = dirname + indexes[i]


if not os.path.exists('./%s' % dirname):
 os.mkdir('./%s' % dirname)

key = '2134_43'
#x = np.arange(10)

#np.save('./%s/%s' % (dirname, key), x)


with open('./%s/%s.npy' % (dirname, key), 'rb') as f:
 x = np.load(f)
print(x)



