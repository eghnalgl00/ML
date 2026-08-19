import torch
import torch.nn as nn
import torch.optim as optim

data = [[1,2,3],[4,5,6]]
my_tensor = torch.tensor(data)

shape = (3,4)

ones = torch.ones(shape)
zeros = torch.zeros(shape)
random = torch.randn(shape)


rand_like = torch.randn_like(my_tensor , dtype = torch.float) # same shape with my_tensor , random values

rand_like , rand_like.shape , rand_like.dtype , rand_like.device

w = torch.tensor([[1.0], [2.0]], requires_grad = True) #every step w takes is tracked , so backprop available

my_tensor.requires_grad # False
w.requires_grad #True


a = torch.tensor(2.0, requires_grad = True)
b = torch.tensor(3.0, requires_grad = True)
x = torch.tensor(4.0, requires_grad = True)

y = a + b
z = torch.sqrt(x * y)

z.grad_fn #SqrtBackward
y.grad_fn #AddBackward
a.grad_fn #None


x1 = torch.tensor([[1,2],[3,4]])
x2 = torch.tensor([[10,20],[30,40]])

x1*x2 #Element-wise , [[10,40], [90,160]]
x1@x2 #Dot product , [[70,100], [150,220]]

scores = torch.tensor([[1,20,3],[5,5,-12]] , dtype = torch.float)
score = torch.tensor([[10,20,30], [40,50,60]] , dtype = torch.float)

avg_score = score.mean(dim = 1) # mean of columns -> [20,50]

nums = torch.arange(36, dtype = torch.float).reshape(6,6)

col_2 = nums[:,2]

col_2.mean() == nums.mean(dim=0)[2] #True


best_indices = torch.argmax(scores , dim = 0) # Indices of largest numbers of ROWS

best_scores = torch.max(scores, dim = 0) #Largest numbers of ROWS


data = torch.tensor([[10,11,12,13],[20,21,22,23],[30,31,32,33]])

indices_to_select = torch.tensor([[2],[0]])

selected_values= torch.gather(data , dim=1 , index = indices_to_select) #Gather the values from COLUMNS 

N = 10
D_in = 1
D_out = 1

X = torch.randn(N,D_in)
true_W = torch.tensor([[2.0]])
true_b = torch.tensor(1.0)
y_true = X @ true_W + true_b + torch.randn(N , D_out) * 0.1 #add some noise

"""

def model(X,W,b): #model
    return X@W + b # Linear Regression


W = torch.randn(D_in , D_out , requires_grad = True)
b = torch.randn(1, requires_grad = True)

y_hat = model(X,W,b)
error = y_hat - y_true
sq_error = (y_hat - y_true) ** 2
loss = sq_error.mean() # MSE - Mean Squared Error Loss
print(f"Loss : {loss}")

loss.backward() # go backward and compute the gradients
print(f"radient for W : {W.grad}") #print gradients
print(f"Gradient for b : {b.grad}")
"""
"""
#Hyperparameters
learning_rate , epochs = 0.01 , 100

W, b = torch.randn(1 , 1 , requires_grad = True) , torch.randn(1 , requires_grad = True)


for epoch in range(epochs):
    #Forward Pass and Loss Computation
    y_hat = X @ W + b
    loss = torch.mean((y_hat-y_true)**2)

    #Calculate gradients
    loss.backward()

    #Update Parameters
    with torch.no_grad():
        W -= learning_rate * W.grad ; b -= learning_rate * b.grad

    #Zero Gradients
    W.grad.zero_() ; b.grad.zero_()   

    #Print out the parameters and loss
    if epoch % 10 == 0 or epoch == epochs -1 :
        print(f"{epoch:02d}. step: loss : {loss.item():.4f} | W : {W.item():.4f} | b : {b.item():.3f} ")
"""

# Create a linear layer with parameters automatically generated
linear_layer = nn.Linear(in_features = D_in , out_features = D_out)  

#See the parameters
linear_layer.weight
linear_layer.bias

#Linear Forward Pass
y_hat_nn = linear_layer(X)

#Relu
relu = nn.ReLU()

#GELU -Standard Distribution-
gelu = nn.GELU()

#Softmax
softmax = nn.Softmax(dim = -1)
logits = torch.randn(1,8)
probs = softmax(logits)

#Embedding , word lookup table
vocab_size = 12 # The language has 12 unique words 
embedding_dim = 20 # Every word has 3D embeddings

embedding_layer = nn.Embedding(vocab_size , embedding_dim)

input_ids = torch.tensor([[1]])
word_vectors = embedding_layer(input_ids)
#print(word_vectors)



#LayerNorm , to prevent exploding / vanishing gradients
norm_layer = nn.LayerNorm(normalized_shape = 3)
input_features = torch.tensor([[[1,2,3], [4,5,6]]] ,dtype = torch.float)




#Dropout (to prevent overfitting) randomly zeroes some neurons ONLY DURING TRAINING
dropout_layer = torch.nn.Dropout(p = 0.3)
input_tensor = torch.randn(1,4)

#Activate dropout for training
dropout_layer.train()
output_during_train = dropout_layer(input_tensor)

#Deactivate dropout for evaluation/prediction
dropout_layer.eval()
output_during_eval = dropout_layer(input_tensor)

#print(f"Output in training {output_during_train}")
#print(f"Evaluation :{output_during_eval}")


#Inherit from nn.Module
class LinearRegressionModel(nn.Module):
    def __init__(self,in_features , out_features):
        super().__init__()
        #In the constructor , we DEFINE the layers we will use
        self.linear_layer = nn.Linear(in_features , out_features)

    def forward(self , x):
        #Connect the layers 
        return self.linear_layer(x)

model = LinearRegressionModel(in_features = 1 , out_features = 1)
#print(model)  

#Hyperparameters
learning_rate  , epochs = 0.001 ,100000

#Create an Adam Optimizer. Pass model.parameters() to tell which tensors to manage
optimizer = optim.Adam(model.parameters() , lr = learning_rate)

#Use MSE
loss_fn = nn.MSELoss()

for epoch in range(epochs):
    #Forward Pass
    y_hat = model(X)

    #Calculate Loss
    loss = loss_fn(y_hat , y_true)

    #1.Zero the gradients
    optimizer.zero_grad()
    #2. Compute gradients
    loss.backward()
    #3. Update the parameters
    optimizer.step()