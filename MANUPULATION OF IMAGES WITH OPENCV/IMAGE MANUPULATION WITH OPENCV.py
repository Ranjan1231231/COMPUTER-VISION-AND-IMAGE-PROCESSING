import numpy as np
import matplotlib.pyplot as plt
import cv2

lenna=cv2.imread("lenna.png")
# plt.figure(figsize=(10,10))
# plt.imshow(cv2.cvtColor(baboon,cv2.COLOR_BGR2RGB))
# plt.show()

#USING NON COPY AND COPY METHOD COMPAIRING IT
A=lenna
B=lenna.copy()
lenna[:,:,]=0
# plt.figure(figsize=(10,10))
# plt.subplot(131)
# plt.imshow(cv2.cvtColor(lenna,cv2.COLOR_BGR2RGB))
# plt.title("lenna")
# plt.subplot(132)
# plt.imshow(cv2.cvtColor(A,cv2.COLOR_BGR2RGB))
# plt.title("array A")
# plt.subplot(133)
# plt.imshow(cv2.cvtColor(B,cv2.COLOR_BGR2RGB))
# plt.title("array B")
# plt.show()



#FLIPPING IMAGES
images=cv2.imread("cat.png")
# plt.figure(figsize=(10,10))
# plt.imshow(cv2.cvtColor(images,cv2.COLOR_BGR2RGB))
# plt.show()

#we can cast it to a array and find its shape
width,height,c=images.shape
# print(("width,height,c",width,height,c))

array_flip=np.zeros((width,height,c),dtype=np.uint8)

#flipping using conventional array first to last place method
for i,row in enumerate(images):
    array_flip[width-1-i,:,:]=row
# plt.figure(figsize=(5,5))
# plt.imshow(cv2.cvtColor(array_flip,cv2.COLOR_BGR2RGB))
# plt.show()

# for flipcode in[0,1,-1]:
#     im_flip=cv2.flip(images,flipcode)
#     plt.imshow(cv2.cvtColor(im_flip,cv2.COLOR_BGR2RGB))
#     plt.title("flipcode"+str(flipcode))
#     plt.show()

#USINGN THE ROTATE FUNCTION
im_flip=cv2.rotate(images,0) #0 is the type of the flip
# plt.imshow(cv2.cvtColor(im_flip,cv2.COLOR_BGR2RGB))
# plt.show()

flip = {"ROTATE_90_CLOCKWISE":cv2.ROTATE_90_CLOCKWISE,"ROTATE_90_COUNTERCLOCKWISE":cv2.ROTATE_90_COUNTERCLOCKWISE,"ROTATE_180":cv2.ROTATE_180}
# for key,value in flip.items():
    # plt.subplot(1,2,1)
    # plt.imshow(cv2.cvtColor(images,cv2.COLOR_BGR2RGB))
    # plt.title("original")
    # plt.subplot(1,2,2)
    # plt.imshow(cv2.cvtColor(cv2.rotate(images,value),cv2.COLOR_BGR2RGB))
    # plt.title(key)
    # plt.show()


#CROPPING AN IMAGE
upper=150
lower=400
crop_top=images[upper:lower,:,:]
# plt.figure(figsize=(5,5))
# plt.imshow(cv2.cvtColor(crop_top,cv2.COLOR_BGR2RGB))
# plt.show()
#

left=150
right=400
crop_horizontal=crop_top[:,left:right,:]
# plt.figure(figsize=(5,5))
# plt.imshow(cv2.cvtColor(crop_horizontal,cv2.COLOR_BGR2RGB))
# plt.show()


#CHANGING SPECIFIC PIXELS
array_sq=np.copy(images)
array_sq[upper:lower,left:right,:]=0

#compairing both results
# plt.figure(figsize=(10,10))
# plt.subplot(1,2,1)
# plt.imshow(cv2.cvtColor(images,cv2.COLOR_BGR2RGB))
# plt.title("orignal")
# plt.subplot(1,2,2)
# plt.imshow(cv2.cvtColor(array_sq,cv2.COLOR_BGR2RGB))
# plt.title("Altered image")
# plt.show()

#We can also create shapes and OpenCV, we can use the method rectangle. The parameter pt1 is the top-left coordinate of the rectangle: (left,top) or  (𝑥0,𝑦0)
 # , pt2 is the bottom right coordinate(right,lower) or  (𝑥1,𝑦1)
 # . The parameter color is a tuple representing the intensity of each channel ( blue, green, red). Finally, we have the line thickness.


#CREATING A RECTANGLE
start_point,end_point=(left,upper),(right,lower)
# image_draw=np.copy(images)
# cv2.rectangle(image_draw,pt1=start_point,pt2=end_point,color=(0,255,0),thickness=3)
# plt.figure(figsize=(5,5))
# plt.imshow(cv2.cvtColor(image_draw,cv2.COLOR_BGR2RGB))
# plt.show()

#We can overlay text on an image using the function putText with the following parameter values:
# img: Image array
# text: Text string to be overlayed
# org: Bottom-left corner of the text string in the image
# fontFace: tye type of font
# fontScale: Font scale
# color: Text color
# thickness: Thickness of the lines used to draw a text
# lineType: Line type


image_draw=cv2.putText(img=images,text="Cat",org=(10,500),color=(255,255,255),fontFace=4,fontScale=5,thickness=2)
# plt.figure(figsize=(10,10))
# plt.imshow(cv2.cvtColor(image_draw,cv2.COLOR_BGR2RGB))
# plt.show()
