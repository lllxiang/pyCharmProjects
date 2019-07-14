# 根据[RGB]顺序，产生cmap25.npy数据和对应的bmp 文件。
# 用于快速比对 标注单位提供的 标注结果是否正确。

import cv2, os
import numpy as np

cmap = np.array([[0,0,0],
[0,255,0],
[255,0,0],
[0,0,255],
[255,255,0],
[255,0,255],
[0,255,255],
[100,100,0],
[100,0,100],
[0,100,100],
[20,20,100],
[20,100,20],
[100,20,20],
[255,100,0],
[255,0,100],
[200,200,100],
[150,100,100],
[100,150,100],
[100,100,150],
[120,25,15],
[25,120,15],
[40,100,200],
[100,40,200],
[18,255,100],
[18,100,100]])

print(cmap.shape)
im = np.zeros([25*20,200,3],dtype=np.uint8)
os.chdir('../python-tools/about-gen-data')
print('now dir=', os.getcwd())

for i in range(25):
    im[i*25:(i+1)*25-1,:,:] = cmap[i,:]

np.transpose(im,(0,2,1))

cv2.imshow('im',im)
cv2.waitKey(0)
cv2.imwrite('cmap.bmp',im)
np.save('camp25.npy',cmap)