import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from tensorboardX import SummaryWriter


def produce_data(batch_size=128, disturb=0.01):
    a = np.random.uniform(10, 100, size=(batch_size,6))
    b = np.random.uniform(10, 100, size=(batch_size,2))

    for i in range(batch_size):
        alpha = np.random.uniform(0.03, 0.1, size=1)
        xstart = np.random.uniform(498, 506, size=1)
        a[i,0] = np.random.uniform(480, 498, size=1)
        a[i,1] = (xstart-a[i,0])*(alpha+np.random.uniform(-disturb, disturb, size=1))
        xgap = np.random.uniform(30, 50, size=1)
        a[i,2] = a[i,0]-xgap
        a[i,3] = (xstart-a[i,2])*(alpha+np.random.uniform(-disturb, disturb, size=1))
        xgap = np.random.uniform(30, 50, size=1)
        a[i,4] = a[i,2]-xgap
        a[i,5] = (xstart-a[i,4])*(alpha+np.random.uniform(-disturb, disturb, size=1))
        b[i] = [1,xstart * alpha]

    a = torch.FloatTensor(a).to(device)
    b = torch.FloatTensor(b).to(device)

    return a, b

# torch.manual_seed(1)    # reproducible

# torch can only train on Variable, so convert them to Variable
# The code below is deprecated in Pytorch 0.4. Now, autograd directly supports tensors

def weights_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        torch.nn.init.constant_(m.bias, 0.01)

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden1, n_hidden2, n_hidden3, n_output):
        super(Net, self).__init__()
        self.hidden1 = torch.nn.Linear(n_feature, n_hidden1)   # hidden layer
        self.hidden2 = torch.nn.Linear(n_hidden1, n_hidden2)
        self.hidden3 = torch.nn.Linear(n_hidden2, n_hidden3)
        self.predict = torch.nn.Linear(n_hidden3, n_output)   # output layer

        self.apply(weights_init)

    def forward(self, x):
        x = F.relu(self.hidden1(x))      # activation function for hidden layer
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = self.predict(x)             # linear output
        return x
writer = SummaryWriter(log_dir="./log")
net = Net(n_feature=6, n_hidden1=64, n_hidden2=128, n_hidden3=64, n_output=2).to(device)     # define the network
print(net)  # net architecture

optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss

plt.ion()   # something about plotting
a,b = produce_data()
for t in range(200000):
    
    prediction = net(a)     # input x and predict based on x

    loss = loss_func(prediction, b)     # must be (1. nn output, 2. target)

    optimizer.zero_grad()   # clear gradients for next train
    loss.backward()         # backpropagation, compute gradients
    optimizer.step()        # apply gradients
    # print('loss = %.4f' % loss.detach().cpu().numpy())
    writer.add_scalar('loss', loss.detach().cpu().numpy(), t) 
torch.save(net.state_dict(), "./state_dict.pt")