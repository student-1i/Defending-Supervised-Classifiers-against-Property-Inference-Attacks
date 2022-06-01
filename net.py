###Define the model class
class Dedend_Net(nn.Module):
    def __init__(self):
        super(Dedend_Net, self).__init__()
        self.fc1 = nn.Linear(41, 32)
        torch.nn.init.xavier_uniform_(self.fc1.weight)

        self.fc2 = nn.Linear(32, 16)
        torch.nn.init.xavier_uniform_(self.fc2.weight)

        self.fc3 = nn.Linear(16, 8)
        torch.nn.init.xavier_uniform_(self.fc3.weight)

        self.fc4 = nn.Linear(8, 2)
        torch.nn.init.xavier_uniform_(self.fc4.weight)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = F.log_softmax(self.fc4(x), dim=1)
        return x
