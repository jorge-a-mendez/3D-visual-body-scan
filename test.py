import numpy as np
import cv2
import detection as dt
from matplotlib import pyplot as plt 


img = []

fgbg = cv2.BackgroundSubtractorMOG2()

i = 0

while(i <= 350):

    # Our operations on the frame come here

    title = "./testimages/image%.3d.jpg" % (i)
    gray = cv2.imread(title, 0)

    img.append(gray)

    # fgmask = fgbg.apply(gray)       # foreground mask

    # img.append(fgmask)

    i = i + 8

    # Display the resulting frame
    # cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


for f in img:
    f = cv2.GaussianBlur(f,(5,5),0)
    contours = dt.contour(f)
    cv2.drawContours(f,contours, -1, (0,255,0), 3)
    f=cv2.resize(f, None, fx=0.45, fy=0.45, interpolation=cv2.INTER_AREA)
    # cv2.imshow('contours',f)
    # cv2.waitKey(25)
    # cv2.imshow('saved', f)
    # cv2.waitKey(25)

# th = dt.thresh(img[2])
# ed = dt.edges(img[2])
# plt.imshow(ed, cmap = 'gray')
# plt.show()
# When everything done, release the capture
# cap.release()
# cv2.destroyAllWindows()




# idx = 0
# for cnt in contours:
#     idx += 1
#     print idx
#     x,y,w,h = cv2.boundingRect(cnt)
#     roi = img[1][y:y+h,x:x+w]
#     cv2.rectangle(img[1], (x,y), (x+w,y+h), (200,0,0),2)
#     plt.imshow(img[1], cmap = 'gray')
#     cv2.waitKey(1000)



CNT_LNGTH_THRSH = 300
mask = np.ones(img[1].shape[:2], dtype="uint8") * 255

for f in img:
    
    contours, person = dt.person(f, CNT_LNGTH_THRSH)

    cv2.imshow("contourcitos",person)
    cv2.waitKey(100)