#GEOMETRIC OPERATIONS AND OTHER MATHEMATICAL TOOLS IN PILLOW
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from PIL import ImageOps

#DEFINING THE HELPER FUNCTION TO PLOT TWO IMAGE SIDE BY SIDE
def plot_image(image_1,image_2,title_1="Orignal",title_2="New_image"):
    plt.figure(figsize=(10,10))
    plt.subplot(1,2,1)
    plt.imshow(image_1,cmap="gray")
    plt.title(title_1)
    plt.subplot(1,2,2)
    plt.imshow(image_2,cmap="gray")
    plt.title(title_2)
    plt.show()

#GEOMETRIC TRANSFORMATIONS
    #it allows us to perform diffrent operations like translation , shift , reshape and rotate the image
image=Image.open("lenna.png")
# plt.imshow(image)
# plt.show()

#scaling the horizontal axis by two
width,height=image.size
# new_width=2*width
# new_height=height
# new_image=image.resize((new_width,new_height))
# plt.imshow(new_image)
# plt.show()


#scaling the vertical axis by two
# new_width=width
# new_height=2*height
# new_image=image.resize((new_width,new_height))
# plt.imshow(new_image)
# plt.show()

#doubling both the width adn the height of the image
# new_width2=2*width
# new_height2=2*height
# new_image3=image.resize((new_width2,new_height2))
# plt.imshow(new_image3)
# plt.show()

#shring the image width and height
# new_width1=width//2
# new_height1=height//2
# new_image2=image.resize((new_width1,new_height1))
# plt.imshow(new_image2)
# plt.show()


#ROTATION
theta=45
# new_image=image.rotate(theta)
# plt.imshow(new_image)
# plt.show()

#MATHEMATICAL OPREATIONS
#ARRAY OPERATIONS
# We can perform array operations on an image; Using Python broadcasting, we can add a constant to each pixel's intensity value.
# Before doing that, we must first we convert the PIL image to a numpy array.
image=np.array(image)
# new_image=image+20
# plt.imshow(new_image)
# plt.show()


# new_image=10*image
# plt.imshow(new_image)
# plt.show()


#We can add the elements of two arrays of equal shape. In this example, we generate an array of random noises with the same shape and data type as our image.

noise=np.random.normal(0,20,(height,width,3)).astype(np.uint8)
# print(noise.shape)


#We add the generated noise to the image and plot the result. We see the values that have corrupted the image:
# new_image=image+noise
# plt.imshow(new_image)
# plt.show()

#At the same time, we can multiply the elements of two arrays of equal shape. We can multiply the random image and the Lenna image and plot the result.
# new_image=image*noise
# plt.imshow(new_image)
# plt.show()


#MATRIX OPERATIONS

#grayscale images are matrixes
im_gray=Image.open("barbara.png")
im_gray=ImageOps.grayscale(im_gray)
im_gray=np.array(im_gray)
# plt.imshow(im_gray,cmap="gray")
# plt.show()

#applying the singular value decompsiton
U,s,V=np.linalg.svd(im_gray,full_matrices=True)
# print(s.shape)
S=np.zeros((im_gray.shape[0],im_gray.shape[1]))
S[:image.shape[0],:image.shape[0]]=np.diag(S)
# plot_image(U,V,title_1="Matrix U",title_2="Matrix V")
# plt.imshow(S,cmap="gray")
# plt.show()


B=S.dot(V)
# plt.imshow(B,cmap="gray")
# plt.show()
A=U.dot(B)
# plt.imshow(A,cmap="gray")
# plt.show()




for n_component in [1,10,100,200,500]:
    S_new=S[:,:n_component]
    V_new=V[:n_component,:]
    A=U.dot(S_new.dot(V_new))
    plt.imshow(A,cmap='gray')
    plt.title("Number of components:"+str(n_component))
    plt.show()
