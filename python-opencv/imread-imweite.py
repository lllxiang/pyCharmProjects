import numpy as np
import cv2
from PIL import Image


la = [[1,2,3],[4,5,6]]
la_np = np.array(la) #list to ndarray

print(np.min(la_np, axis=0))    #列    N C H,W
print(np.min(la_np, axis=1))

im = cv2.imread('/home/lx/data/demo.jpg',cv2.IMREAD_GRAYSCALE)
cv2.imshow('im_src',im)
k = cv2.waitKey(0)
if k == 27:
    cv2.destroyAllWindows()
elif k==ord('s'):
    cv2.imwrite('dst.jpg',im)
    cv2.waitKey(10)
    cv2.destroyAllWindows()


im2 = np.array(Image.open('dst.jpg'))
print('im2.shape', im2.shape)