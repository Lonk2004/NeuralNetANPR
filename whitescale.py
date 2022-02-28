import matplotlib.pyplot as plt 
import cv2
import numpy as np 

img = cv2. imread(r'D:\Downloads\archive\imagesextrct\cars6.png')
res = cv2. resize(img, dsize=(150, 32), interpolation=cv2. INTER_CUBIC)
cv2.imshow("image", res)
cv2.waitKey(3000)
cv2.destroyAllWindows()
def whitescale(rgb):
    h, w, c = 150, 32, 3
    X =  np.zeros((w,h,c))
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    for x in range (0,31):
        for y in range(0,149): 
            for z in range(0,3):
                if r[x,y] < 70 and g[x,y] < 70 and b[x,y] < 50:
                    X[x,y,z] = 255
                elif r[x,y] < 100 and g[x,y] < 100 and b[x,y] < 100:
                    X[x,y,z] = 127
                else:
                    X[x,y,z] = 0
    return X

def Detstatus(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    x, y = 3, 100
    colourfound = False
    while colourfound == False: 
        if (b[x][y] > 100):
            return ("white")
        elif (r[x][y] < 100) and (g[x][y] < 100):
            y += 1
        else: 
            return ("yellow")


X = np.asarray(whitescale(res)) 
print(Detstatus(res))
cv2.imshow("image", X)
cv2.waitKey(3000)
cv2.destroyAllWindows()



