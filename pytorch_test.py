import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


class Model(nn.Module):
    def __init__(self,features=4,h1=8,h2=9,out_features=3):
        super().__init__()
        self.fc1=nn.Linear(features,h1)
        self.fc2 = nn.Linear(h1,h2)
        self.out = nn.Linear(h2,out_features)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x
    


random_seed = 41
torch.manual_seed(random_seed)

model = Model()
print(model)


url = "https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv"
my_df = pd.read_csv(url)

#prepocess the data
my_df["variety"] = my_df["variety"].replace("Setosa",0)
my_df["variety"] = my_df["variety"].replace("Versicolor",1)
my_df["variety"] = my_df["variety"].replace("Virginica",2)
print(my_df)


##split into training and testing data (this level the data is in numpy arrays)
X = my_df.drop("variety",axis=1).values #the data to be fed into the neural network
y = my_df["variety"].values #the correct classification


scaler = StandardScaler()
X = scaler.fit_transform(X)



#this level split the data and convert to tensors (numpy arrays)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=random_seed)
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

# print(X_train)


#measure loss
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.01) #behind the scenes way to optimize weights and biases during back propagation


epochs = 100
losses = []

for epoch in range(epochs):
    y_pred = model.forward(X_train)

    loss = criterion(y_pred,y_train) #calculates loss the models predicted value vs the actual value in the dataset
    losses.append(loss.detach().numpy())

    if epoch % 10 == 0:
        print(f"Epoch: {epoch} and loss: {loss}")

    #back propagation (takes the predicted value and feeds back into the network until the max number of epochs reached )
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

torch.save(model.state_dict(),"iris_model.pt")
joblib.dump(scaler, 'scaler.joblib')

plt.plot(range(epochs),losses)
plt.ylabel("error")
plt.xlabel("epoch")
plt.show()



#test the model
with torch.no_grad(): #turn off backpropagation
     y_eval = model.forward(X_test)
     loss = criterion(y_eval,y_test) #calculates the accuracy of the models predicted value vs the actual value in the dataset

# print(loss)

correct = 0
with torch.no_grad():
    for i,data in enumerate(X_test):
        y_val = model.forward(data) #returns 3 values in a list the higher the number the higher the model thinks its that classification
        #0 -> setosa 1-> versicolor 2-> virginica


        print(f"{i+1} -> {str(y_val)} the real answer: {y_test[i]}") 

        if y_val.argmax().item() == y_test[i]:
            correct += 1


print(f"the model got {correct} correct out of {len(y_test)}")

