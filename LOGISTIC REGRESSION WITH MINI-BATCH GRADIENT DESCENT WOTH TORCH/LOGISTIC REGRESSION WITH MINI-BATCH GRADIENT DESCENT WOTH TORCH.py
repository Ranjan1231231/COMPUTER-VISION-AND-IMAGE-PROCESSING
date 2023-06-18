#LOGISTIC REGRESSION WITH MINI-BATCH GRADIENT DESCENT WOTH TORCH
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import torch
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn
# Creating class for plotting and the function for plotting
class plot_error_surfaces(object):
    #Construstor
    def __init__(self,w_range,b_range,X,Y,n_samples=30,go=True):
        W=np.linspace(-w_range,w_range,n_samples)
        B=np.linspace(-b_range,b_range,n_samples)
        w,b=np.meshgrid(W,B)
        Z=np.zeros((30,30))
        count1=0
        self.y=Y.numpy()
        self.x=X.numpy()
        for w1,b1 in zip(w,b):
            count2=0
            for w2,b2 in zip(w1,b1):
                yhat=1/(1+np.exp(-1*(w2*self.x+b2)))
                Z[count1,count2]=-1*np.mean(self.y*np.log(yhat+1e-16)+(1-self.y)*np.log(1-yhat+1e-16))
                count2+=1
            count1+=1
        self.Z=Z
        self.w=w
        self.b=b
        self.W=[]
        self.B=[]
        self.LOSS=[]
        self.n=0
        if go==True:
            plt.figure(figsize = (7.5,5))
            plt.axes(projection='3d').plot_surface(self.w,self.b,self.Z,rstride = 1,cstride = 1,cmap = 'viridis',edgecolor='none')
            plt.title("Loss surface")
            plt.xlabel("w")
            plt.ylabel("b")
            plt.show()
            plt.figure()
            plt.title("loss surface contour")
            plt.xlabel("w")
            plt.ylabel("b")
            plt.contour(self.w,self.b,self.Z)
            plt.show()
    #setter
    def set_para_loss( self,model,loss ):
        self.n=self.n+1
        self.W.append(list(model.parameters())[0].item())
        self.B.append(list(model.parameters())[1].item())
        self.LOSS.append(loss)
    #plotdiagram
    def final_plot( self ):
        ax=plt.axes(projections='3d')
        ax.plot_wireframe(self.w,self.b,self.Z)
        ax.scatter(SElf.W,self.B,self.LOSS,c='r',marker='x',s=200,alph=1)
        plt.figure()
        plt.contour(self.w,self.b,self.Z)
        plt.scatter(self.W,self.B,c='r',marker = 'x')
        plt.xlabel("w")
        plt.ylabel("b")
        plt.show()
    #plot diagram
    def plot_ps( self ):
        plt.subplots(121)
        plt.ylim()
        plt.plot(self.x[self.y==0],self.y[self.y==0],'ro',label='training points')
        plt.plot(self.x[self.y==1],self.y[self.y==1]-1,'o',label='training points')
        plt.plot(self.x,self.W[-1]*self.x+self.B[-1],label="estimated line")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.ylim((-0.1,2))
        plt.title("Data Space Iteration : "+str(self.n))
        plt.show()
        plt.subplot(122)
        plt.contour(self.w,self.b,self.Z)
        plt.scatter(self.W,self.B,c="r",marker = 'x')
        plt.title("Loss surface contour iteration"+str(self.n))
        plt.xlabel("w")
        plt.ylabel("b")
#plot the diagram
def PlotStuff(X,Y,model,epoch,leg=True):
    plt.plot(X.numpy(),model(X).detach().numpy(),label=("epoch"+str(epoch)))
    plt.plot(X.numpy(),Y.numpy(),'r')
    if leg==True:
        plt.legend()
    else:
        pass

#set the random seed
torch.manual_seed(0)

#LOAD DATA
#create the custom Data class which inherts Dataset
class Data(Dataset):
    #constructor
    def __init__(self):
        #create x values from -1 to 1 with step .1
        self.x=torch.arange(-1,1,0.1).view(-1,1)
        #create y values all set to 0
        self.y=torch.zeros(self.x.shape[0],1)
        #set the x values above 0.2 to 1
        self.y[self.x[:,0]>0.2]=1
        #set the .len attriute because we need to override the __len__ nethod
        self.len=self.x.shape[0]
    #getter that returns the data at the given index
    def __getitem__( self,index ):
        return  self.x[index],self.y[index]
    #get length of the dataset
    def __len__( self ):
        return  self.len
#create data object
data_set=Data()
# print(data_set.x)
# print(data_set.y)
len(data_set)
# x,y = data_set[0]
# print("x = {},  y = {}".format(x,y))
# x,y = data_set[1]
# print("x = {},  y = {}".format(x,y))
#  We can see we can separate the one-dimensional dataset into two classes:
plt.plot(data_set.x[data_set.y==0],data_set.y[data_set.y==0],'ro',label="y=0")
plt.plot(data_set.x[data_set.y==1],data_set.y[data_set.y==1]-1,'o',label="y=1")
plt.xlabel('x')
plt.legend()


#CREATE THE MODEL AND TOTAL LOSS FUNCTION
#CREATE LOGISTIC_REGRESSION CLASS THAT INHERITS NN.MODULE WHICH IS THE BASE CLASS FOR ALL

class logistic_regression(nn.Module):
    #constructor
    def __init__(self,n_inputs):
        super(logistic_regression,self).__init__()
        #single layer of logisticregression with number of inputs being n_input and there beig 1 output
        self.linear=nn.Linear(n_inputs,1)
    #prediction
    def forward(self,x):
        #using the input x value puts it through the single layer defined above
        yhat=torch.sigmoid(self.linear(x))
        return yhat

#we can chech the number of features an x value has the size of the input or the dimension of x
# x,y=data_set[0]
# print(len(x))
#CREATE THE LOGISTIC REGRESSION RESULT
model=logistic_regression(1)
# x=torch.tensor([-1.0])
# sigma=model(x)
# print(sigma)
# x,y=data_set[2]
sigma=model
# print(sigma)

#CREATE THE PLOT ERROR SURFACE OBJECT
#15 is the range of w
#13 is the range of b
#data_set[:][0] are all the X values
#data_set[:][1] are all the y values
get_surface=plot_error_surfaces(15,13,data_set[:][0],data_set[:][1])
criterion=nn.BCELoss()
x,y=data_set[0]
# print("x={},y={}".format(x,y))

#making pediction using the model
sigma=model(x)
# print(sigma)
loss=criterion(sigma,y)
# print(loss)

#SETTING THE BATCH SIE USING A DATA LOADER
batch_Size=10
trainloader=DataLoader(dataset = data_set,batch_size=10)
dataset_iter=iter(trainloader)
X,y=next(dataset_iter)
# print(x)

#SETTING THE LEARNING RATE
learning_rate=0.1
optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate)

#TRANING THE MODEL VIA MINI BATCH GRADIENT DESCENT
#mini batch gradient descent
#recreating the get surface object again so that for each example we get a loss surface for that model only
get_surface=plot_error_surfaces(15,13,data_set[:][0],data_set[:][1],30)
#train the model
#FIRST WE CREATE AN INSTANCE OF THE MODEL WE WANT TO TRAIN
model=logistic_regression(1)
#we create a criterion which will measure loss
criterion=nn.BCELoss()
#we create an optimizer withthe model parameters and learnign rate
optimizer=torch.optim.SGD(model.parameters(),lr=.01)
#then we set the number of epochs which is the total number of times we will train the entire training dataset
epochs=500
#this will store the loss over iterations so we can plot it at the end
loss_values=[]
#loop will execute for number of epochs
for epoch in range(epochs):
    #for each batch in the traning data
    for x,y in trainloader:
        #make our predictions from the x values
        yhat=model(x)
        #measure the loss between our prediction and actual y values
        loss=criterion(yhat,y)
        #reset the calculated gradient value ,this must be done each time as itaccumulates if we do not reset
        optimizer.zero_grad()
        #calculates the gradient value with respect to each weight ad bias
        loss.backward()
        #updates the weight and bias according to calculated gradient value
        optimizer.step()
        #set the parameter for the loss surface contour graphs
        get_surface.set_para_loss(model,loss.tolist())
        #save the loss of the iteration
        loss_values.append(loss)
    #want to print the data space for the current iteration every 20 epochs
    # if epoch %20==0:
        # get_surface.plot_ps()

#we can see the final values of the veight and bias. this weight and bias correspond to the orange line in the data space graph and the final spot of the x in the loss surface contour graph
w=model.state_dict()['linear.weight'].data[0]
b=model.state_dict()['linear.bias'].data[0]
# print("w=",w,"b=",b)

#getting the prediction
yhat=model(data_set.x)
#rounding the prediction to the nearedt integer 0 or 1 represinting the classes
yhat=torch.round(yhat)
#counter to keep track of correct predictions
correct=0
#goes through each prediction and actual y value
for prediction, actual in zip(yhat,data_set.y):
    #compare if the prediction and actual y values are the same
    if (prediction==actual):
        #adds to counter if prediction is correct
        correct+=1
#outputs the accuracy by dividing the correct prediction by the length of the dataset
# print("accuracy:",correct/len(data_set)*100,"%")

#FINALLY WE PLOT THE COST VS ITERATION GRAPH, ALTHOUGH IT IS DOWNWARD SLOPING
# plt.plot([value.detach().numpy() for value in loss_values])
# plt.xlabel("iteration")
# plt.ylabel("cost")
# plt.show()

#STOCHASTIC GRADIENT DESCENT
get_surface=plot_error_surfaces(15,13,data_set[:][0],data_set[:][1],30)
#training the model
#first we create an instance of the model we want to train
model=logistic_regression(1)
#we createa critetion which will measure loss
criterion=nn.BCELoss()
#we create a data loader with the datasetadne specified batch size of 1
trainloader=DataLoader(dataset = data_set,batch_size = 1)
#we create an optimiser with the model parameters and learning rate
optimizer=torch.optim.SGD(model.parameters(),lr = .01)
#te we set the number of epochs which is the total number of times we will train on the entire traning dataset
epochs=100
# this will store the loss over iterations so we plot it at the end
loss_values=[]
#loop will execute for number of epochs
for epoch in range(epochs):
    #for each batch in the training data
    for x,y in trainloader:
        #make our predictions from the x values
        yhat=model(x)
        #measure the loss between our prediction and actual y values
        loss=criterion(yhat,y)
        #resets the calculated gradient value, this must be done each time as it accumulatesif we dont reset
        optimizer.zero_grad()
        #calculates the gradient value with respect to each weight and bias
        loss.backward()
        #updates the weight and bias according to calculated gradient value
        optimizer.step()
        #set the parameters for the loss surface contour graphs
        get_surface.set_para_loss(model,loss.tolist())
        #saves the loss of the iteration
        loss_values.append(loss)
    # if epoch % 20==0:
        # get_surface.plot_ps()
w=model.state_dict()['linear.weight'].data[0]
b=model.state_dict()['linear.bias'].data[0]
print("w=",w,"b=",b)
#getting the predictions
yhat=model(data_set.x)
#rounding the prediction to the nearest integer 0 or 1 representing the classes
yhat=torch.round(yhat)
#counter to keep track of correct predictions
correct=0
#goes through each prediction and actual y value
for prediction, actual in zip(yhat,data_set.y):
    #compares if the prediction and actualy y values are the same
    if(prediction==actual):
        #adds to counter if prediction is correct
        correct+=1
#outputs the accuracy by dividing the correct predictions by the length of the dataset
print("Accuracy:",correct/len(data_set)*100,"%")
# plt.plot([value.detach().numpy() for value in loss_values])
# plt.xlabel("itertion")
# plt.ylabel("cost")
# plt.show()



#HIGH LEARNING RATE
get_surface=plot_error_surfaces(15,13,data_set[:][0],data_set[:][1],30)
#traning the model
#first we create an instance of the model we want to train
model=logistic_regression(1)
#we create a criterion tha twill measure loss
criterion=nn.BCELoss()
#we create a data loader with the dataset nd specified batch size of 1
trainloader=DataLoader(dataset = data_set,batch_size = 1)
#we create an optimizer with the model parameters and learning rate
optimizer=torch.optim.SGD(model.parameters(),lr=1)
#then we set the number of epochs which is the total number of times we will train on the entire training dataset
epochs=100
#this will store the loss over iterations so we canplot it at tht end
loss_values=[]

# Loop will execute for number of epochs
for epoch in range(epochs):
    # For each batch in the training data
    for x, y in trainloader:
        # Make our predictions from the X values
        yhat = model(x)
        # Measure the loss between our prediction and actual Y values
        loss = criterion(yhat, y)
        # Resets the calculated gradient value, this must be done each time as it accumulates if we do not reset
        optimizer.zero_grad()
        # Calculates the gradient value with respect to each weight and bias
        loss.backward()
        # Updates the weight and bias according to calculated gradient value
        optimizer.step()
        # Set the parameters for the loss surface contour graphs
        get_surface.set_para_loss(model, loss.tolist())
        # Saves the loss of the iteration
        loss_values.append(loss)
    # Want to print the Data Space for the current iteration every 20 epochs
    # if epoch % 20 == 0:
    #     get_surface.plot_ps()
w = model.state_dict()['linear.weight'].data[0]
b = model.state_dict()['linear.bias'].data[0]
print("w = ", w, "b = ", b)
# Getting the predictions
yhat = model(data_set.x)
# Rounding the prediction to the nearedt integer 0 or 1 representing the classes
yhat = torch.round(yhat)
# Counter to keep track of correct predictions
correct = 0
# Goes through each prediction and actual y value
for prediction, actual in zip(yhat, data_set.y):
    # Compares if the prediction and actualy y value are the same
    if (prediction == actual):
        # Adds to counter if prediction is correct
        correct+=1
# Outputs the accuracy by dividing the correct predictions by the length of the dataset
print("Accuracy: ", correct/len(data_set)*100, "%")
plt.plot([value.detach().numpy() for value in loss_values])
plt.xlabel("Iteration")
plt.ylabel("Cost")

#QUESTION
#Using the following code train the model using a learning rate of .01, 120 epochs, and batch_size of 1.
get_surface = plot_error_surfaces(15, 13, data_set[:][0], data_set[:][1], 30)
#Train the Model
# First we create an instance of the model we want to train
model = logistic_regression(1)
# We create a criterion which will measure loss
criterion = nn.BCELoss()
# We create a data loader with the dataset and specified batch size of 1
trainloader = DataLoader(dataset = data_set, batch_size = batch_Size)
# We create an optimizer with the model parameters and learning rate
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
# Then we set the number of epochs which is the total number of times we will train on the entire training dataset
epochs= 100
# This will store the loss over iterations so we can plot it at the end
loss_values = []

# Loop will execute for number of epochs
for epoch in range(epochs):
    # For each batch in the training data
    for x, y in trainloader:
        # Make our predictions from the X values
        yhat = model(x)
        # Measure the loss between our prediction and actual Y values
        loss = criterion(yhat, y)
        # Resets the calculated gradient value, this must be done each time as it accumulates if we do not reset
        optimizer.zero_grad()
        # Calculates the gradient value with respect to each weight and bias
        loss.backward()
        # Updates the weight and bias according to calculated gradient value
        optimizer.step()
        # Set the parameters for the loss surface contour graphs
        get_surface.set_para_loss(model, loss.tolist())
        # Saves the loss of the iteration
        loss_values.append(loss)
    # Want to print the Data Space for the current iteration every 20 epochs
    # if epoch % 20 == 0:
    #     get_surface.plot_ps()
w = model.state_dict()['linear.weight'].data[0]
b = model.state_dict()['linear.bias'].data[0]
print("w = ", w, "b = ", b)
# Getting the predictions
yhat = model(data_set.x)
# Rounding the prediction to the nearedt integer 0 or 1 representing the classes
yhat = torch.round(yhat)
# Counter to keep track of correct predictions
correct = 0
# Goes through each prediction and actual y value
for prediction, actual in zip(yhat, data_set.y):
    # Compares if the prediction and actualy y value are the same
    if (prediction == actual):
        # Adds to counter if prediction is correct
        correct+=1
# Outputs the accuracy by dividing the correct predictions by the length of the dataset
print("Accuracy: ", correct/len(data_set)*100, "%")
plt.plot([value.detach().numpy() for value in loss_values])
plt.xlabel("Iteration")
plt.ylabel("Cost")