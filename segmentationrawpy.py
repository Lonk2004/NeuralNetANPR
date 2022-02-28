from whitescale import X 

import numpy as np 
import cv2 

def segmentation(whiimg):
    h, w, c= 15, 32, 3
    chimg =  np.zeros((w,h,c))
    i = 5
    b = 0
    while i < 140:
        for x in range(2,30): 
            if ((whiimg[x][i] == 255)):
                for j in range (0,14):
                    for k in range (0,31):
                        chimg[k,j,0] = whiimg[k][i+j] 
                        chimg[k,j,1] = whiimg[k][i+j] 
                        chimg[k,j,2] = whiimg[k][i+j] 
                cv2.imshow("image", chimg)
                cv2.waitKey(3000)
                cv2.destroyAllWindows()
                newpath = r"D:\Downloads\archive\SegmentsFromPlate\Seg" + str(b) +'.png'
                cv2.imwrite(newpath, chimg)
                i += 15
                x = 4
                b += 1
        i += 1
X_values = X[:,:,1]
segmentation(X_values)
num_chars = b+1