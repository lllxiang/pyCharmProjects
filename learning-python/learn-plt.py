# -*- coding: utf-8 -*-
#! /bin/python
import numpy as np
import matplotlib.pyplot as plt
import scipy
img = np.zeros((200,200),dtype=np.uint8)
img[50:100,50:100] = 100
plt.figure()
plt.subplot(211); plt.imshow(img)
plt.subplot(212); plt.imshow(img,cmap='gray')
scipy.misc.imsave('test.jpg', img)
