import cv2
import numpy as np



# mouse callback function
def draw_rectangle(event,x,y,flags,param):
    global ix,iy,drawing,ex,ey,dst,img

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            dst[:] = img[:]
            cv2.rectangle(dst,(ix,iy),(x,y),(255),1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        ex = x
        ey = y
        dst[:] = img[:]
        cv2.rectangle(dst,(ix,iy),(x,y),(255),1)
        
def MouseInit(image):
    global img,dst,drawing,ix,iy,ex,ey
    drawing = False # true if mouse is pressed
    ix,iy = -1,-1
    ex,ey = -1,-1
    img = np.zeros((500,1000))
    img = image
    if(image.shape[0] > 1000):            
        img = cv2.resize(image,(1000,500))
        
    dst = np.zeros_like(img)
    dst[:] = img[:] 
    cv2.namedWindow('image')
    cv2.setMouseCallback('image',draw_rectangle)
    while(1):
        cv2.imshow('image',dst)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
    cv2.destroyAllWindows()
    x1 = ix*image.shape[1]/img.shape[1]
    y1 = iy*image.shape[0]/img.shape[0]
    x2 = ex*image.shape[1]/img.shape[1]
    y2 = ey*image.shape[0]/img.shape[0]
    return (int(x1),int(y1)),(int(x2),int(y2))

#[x1,y1,x2,y2]=MouseInit(img1)