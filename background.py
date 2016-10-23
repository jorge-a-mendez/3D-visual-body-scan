import cv2
import numpy as np
from matplotlib import pyplot as plt
 
f = cv2.imread("image000.jpg")
bg = np.float32(f)
det_obj = []
img = []
i = 5
while(i <= 241):
    title = "image%.3d.jpg" % (i)
    print title
    f = cv2.imread(title)
    img.append(f)
    f = np.float32(f)
    cv2.accumulateWeighted(f,bg, 0.15)
    res = f - bg

    res = cv2.convertScaleAbs(res)
    


    det_obj.append(res)
    #img.append(f)
    i += 5



# for f in det_obj:
#     cv2.imshow('saved', f)
#     cv2.waitKey(100)
print det_obj[20]
plt.subplot(1,2,1)
plt.imshow(cv2.Canny(det_obj[30],50,100))
plt.subplot(1,2,2)
plt.imshow(img[30])
plt.show()

