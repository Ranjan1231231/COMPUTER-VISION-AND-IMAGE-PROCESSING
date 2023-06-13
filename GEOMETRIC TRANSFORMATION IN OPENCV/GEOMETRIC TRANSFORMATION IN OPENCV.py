#GEOMETRIC TRANSFORMATION USING OPENCV


import matplotlib.pyplot as plt
import cv2
import numpy as np

#HELPER FUNCTION TO PLOT TWO IMAGES SIDE BY SIDE
def plot_image(image_1,image_2,title_1="Original",title_2="New Image"):
    plt.figure(figsize = (10,10))
    plt.subplot(1,2,1)
    plt.imshow(image_1,cmap = "gray")
    plt.title(title_1)
    plt.subplot(1,2,2)
    plt.imshow(image_2,cmap = "gray")
    plt.title(title_2)
    plt.show()


#Geometric Transformations
toy_image=np.zeros((10,10))
toy_image[1:9,1:9]=255
toy_image[2:8,2:8]=0
toy_image[3:7,3:7]=255
toy_image[4:6,4:6]=0
# plt.imshow(toy_image,cmap = "gray")
# plt.show()

#The parameter interpolation estimates pixel values based on neighboring pixels. INTER_NEAREST uses the nearest pixel and INTER_CUBIC uses several pixels near the pixel value we would like to estimate.
new_toy=cv2.resize(toy_image,None,fx=2,fy=1,interpolation=cv2.INTER_NEAREST)
# plt.imshow(new_toy,cmap = "gray")
# plt.show()


image=cv2.imread("lenna.png")
# plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
# plt.show()

#SCALING THE HORIZONTAL AXIS BY 2 AND LEAVING THE VERTICAL AXIS AS IT IS
# new_image=cv2.resize(image,None,fx=2,fy=1,interpolation=cv2.INTER_CUBIC)
# plt.imshow(cv2.cvtColor(new_image,cv2.COLOR_BGR2RGB))
# plt.show()
# print("OLD IMAGE SHAPE : ",image.shape,"new image shape :",new_image.shape)

#in the same manner, scaling the vertial axis by two
# new_image=cv2.resize(image,None,fx=1,fy=2,interpolation=cv2.INTER_CUBIC)
# plt.imshow(cv2.cvtColor(new_image,cv2.COLOR_BGR2RGB))
# plt.show()
# print("OLD IMAGE SHAPE :",image.shape,"new image shape :",new_image.shape)

#scaling both images by 2
# new_image=cv2.resize(image,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC)
# plt.imshow(cv2.cvtColor(new_image,cv2.COLOR_BGR2RGB))
# plt.show()
# print("old image shape:", image.shape, "new image shape:", new_image.shape)


#shrinking the image by seting the scaling factor to a real number between 0 and 1
# new_image=cv2.resize(image,None,fx=1,fy=0.5,interpolation=cv2.INTER_CUBIC)
# plt.imshow(cv2.cvtColor(new_image,cv2.COLOR_BGR2RGB))
# plt.show()
# print("old image shape:", image.shape, "new image shape:", new_image.shape)


#we can also specify the number of rows and columns
# rows=100
# cols=200
# new_image=cv2.resize(image,(100,200),interpolation=cv2.INTER_CUBIC)
# plt.imshow(cv2.cvtColor(new_image,cv2.COLOR_BGR2RGB))
# plt.show()
# print("old image shape:", image.shape, "new image shape:", new_image.shape)




#TRANSLATION
tx=100
ty=0
M=np.float32([[1,0,tx],[0,1,ty]])
# print(M)
rows,cols,_=image.shape
#We use the function warpAffine from the cv2 module. The first input parater is an image array, the second input parameter is the transformation matrix M, and the final input paramter is the length and width of the output image  (𝑐𝑜𝑙𝑠,𝑟𝑜𝑤𝑠)
# new_image=cv2.warpAffine(image,M,(cols,rows))
# plt.imshow(cv2.cvtColor(new_image,cv2.COLOR_BGR2RGB))
# plt.show()

# We can see some of the original image has been cut off. We can fix this by changing the output image size: (cols + tx,rows + ty):
new_image=cv2.warpAffine(image,M,(cols+tx,rows+ty))
# plt.imshow(cv2.cvtColor(new_image,cv2.COLOR_BGR2RGB))
# plt.show()
tx = 0
ty = 50
M = np.float32([[1, 0, tx], [0, 1, ty]])
new_iamge = cv2.warpAffine(image, M, (cols + tx, rows + ty))
# plt.imshow(cv2.cvtColor(new_iamge, cv2.COLOR_BGR2RGB))
# plt.show()