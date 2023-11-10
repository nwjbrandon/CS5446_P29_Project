import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import sys

class Curiosity2:
    def __init__(self, curiosity_vec, train_every=500, mem_capacity=50000, batch_size=32):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        #sys.exit()

        self.curiosity_net = Multisoftmax_ffnn_small(np.sum(curiosity_vec),curiosity_vec[:-1]).to(self.device)
        #self.curiosity_net = Multisoftmax_ffnn(np.sum(curiosity_vec),curiosity_vec[:-1]).to(self.device)
        #self.curiosity_net = Multisoftmax_ffnn_big(np.sum(curiosity_vec),curiosity_vec[:-1]).to(self.device)
        #self.curiosity_net = Multisoftmax_ffnn_bigger(np.sum(curiosity_vec),curiosity_vec[:-1]).to(self.device)
        self.curiosity_net.apply(init_weights)

        self.discrete_obs = curiosity_vec

        self.memory_target = []
        self.memory_prediction = []
        self.train_counter = 1
        self.position = 0
        self.train_every = train_every
        self.mem_capacity = mem_capacity
        self.batch_size = batch_size
        self.marginal_loss = nn.MultiMarginLoss()

    def train(self, current_state, action, next_state, optimizer, eta = 1e-3):

        current_state = self.oneHotEncoding(np.append(current_state,action))
        current_state = torch.from_numpy(current_state).float()

        #compute curiosity reward
        self.curiosity_net.eval()
        current_state = current_state.unsqueeze(0).to(self.device)
        pred_state=self.curiosity_net.forward(current_state)

        #next_state = torch.from_numpy(next_state).float().to(self.device)

        loss = 0
        for i in range(len(pred_state)):
            target = torch.from_numpy(np.array([next_state[i]])).to(self.device)
            #loss += F.cross_entropy(input=pred_state[i],target=target)
            #loss += self.marginal_loss(pred_state[i], target)

            #Hand crafted loss. +1 for every missclasification in the one-hot encoding
            if np.argmax(pred_state[i].cpu().detach().numpy()) != target.cpu().detach().numpy()[0]:
                loss += 1
        loss = eta * loss

        #save states into memory buffer
        self.push(next_state, current_state)

        if (self.train_counter % self.train_every) == 0 and (len(self.memory_target) >= self.batch_size):
            batch_mask = self.sample_index(self.batch_size)
            loss_batch = 0
            self.curiosity_net.train()
            optimizer.zero_grad()
            current_batch = [self.memory_prediction[i] for i in batch_mask]
            current_batch = torch.cat(current_batch)
            current_batch = torch.reshape(current_batch, (self.batch_size,-1))

            prediction_batch = self.curiosity_net.forward(current_batch)

            for j in range(len(next_state)):

                prediction_batch_sub = prediction_batch[j]

                prediction_batch_sub = torch.reshape(prediction_batch_sub, (self.batch_size,-1))

                target_batch = [torch.from_numpy(np.array([self.memory_target[i][j]])).to(self.device) for i in batch_mask]
                target_batch = torch.cat(target_batch)
                #target_batch = torch.reshape(target_batch, (self.batch_size))
                #print(target_batch.size())
                #print(target_batch)
                #print(prediction_batch.size())
                #print(prediction_batch)

                loss_batch += F.cross_entropy(input=prediction_batch_sub,target=target_batch)
                #loss_batch += self.marginal_loss(prediction_batch_sub, target_batch)
            loss_batch = (1/self.batch_size) * loss_batch

            #print('TRAINED')
            #print('LOSS: ', loss_batch)
            loss_batch.backward(retain_graph=True)
            optimizer.step()

        self.train_counter += 1

        #return loss.cpu().detach().numpy()
        #return loss.detach().numpy()
        return loss
    def push(self, next_state, pred_state):
        #If we have space, we add a new slot
        if len(self.memory_target) < self.mem_capacity:
            self.memory_target.append(None)
            self.memory_prediction.append(None)
        self.memory_target[self.position] = next_state
        self.memory_prediction[self.position] = pred_state
        #if we surpass the capacity, we restart the position to 0
        self.position = (self.position + 1) % self.mem_capacity

    def sample_index(self, batch_size):
        return random.sample(range(0,len(self.memory_target)), batch_size)

    def oneHotEncoding(self, observation):
        obs = np.array([])
        for i in range(len(observation)):
            zeros = np.zeros(self.discrete_obs[i])
            zeros[observation[i]] = 1
            obs = np.append(obs,zeros)
        return obs



def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_normal_(m.weight)
            m.bias.data.fill_(0)

class Multisoftmax_ffnn_big(nn.Module):
    def __init__(self, input_size, output_vector):
        #super() gives access to methods in a
        #superclass from the subclass that inherits from it
        super(Multisoftmax_ffnn_big, self).__init__()
        self.lin1 = nn.Linear(input_size, 1024) #
        self.bn1 = nn.BatchNorm1d(1024)
        self.lin2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.lin3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.lin4 = nn.Linear(256, 128)
        self.bn4 = nn.BatchNorm1d(128)

        self.linears = nn.ModuleList([nn.Linear(128,i) for i in output_vector])

    def forward(self, x):
        x = F.relu(self.bn1(self.lin1(x)))
        x = F.relu(self.bn2(self.lin2(x)))
        x = F.relu(self.bn3(self.lin3(x)))
        x = F.relu(self.bn4(self.lin4(x)))

        yhat = [self.linears[i](x) for i,l in enumerate(self.linears)]

        return yhat

class Multisoftmax_ffnn_bigger(nn.Module):
    def __init__(self, input_size, output_vector):
        #super() gives access to methods in a
        #superclass from the subclass that inherits from it
        super(Multisoftmax_ffnn_bigger, self).__init__()
        self.lin1 = nn.Linear(input_size, 2048) #
        self.bn1 = nn.BatchNorm1d(2048)
        self.lin2 = nn.Linear(2048, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.lin3 = nn.Linear(1024, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.lin4 = nn.Linear(512, 256)
        self.bn4 = nn.BatchNorm1d(256)

        self.linears = nn.ModuleList([nn.Linear(256,i) for i in output_vector])

    def forward(self, x):
        x = F.relu(self.bn1(self.lin1(x)))
        x = F.relu(self.bn2(self.lin2(x)))
        x = F.relu(self.bn3(self.lin3(x)))
        x = F.relu(self.bn4(self.lin4(x)))

        yhat = [self.linears[i](x) for i,l in enumerate(self.linears)]

        return yhat

class Multisoftmax_ffnn(nn.Module):
    def __init__(self, input_size, output_vector):
        #super() gives access to methods in a
        #superclass from the subclass that inherits from it
        super(Multisoftmax_ffnn, self).__init__()
        self.lin1 = nn.Linear(input_size, 512) #
        self.bn1 = nn.BatchNorm1d(512)
        self.lin2 = nn.Linear(512, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.lin3 = nn.Linear(128, 32)
        self.bn3 = nn.BatchNorm1d(32)

        self.linears = nn.ModuleList([nn.Linear(32,i) for i in output_vector])
        #self.softmax = nn.ModuleList([nn.Softmax() for i in output_vector])

    def forward(self, x):
        x = F.relu(self.bn1(self.lin1(x)))
        x = F.relu(self.bn2(self.lin2(x)))
        x = F.relu(self.bn3(self.lin3(x)))

        yhat = [self.linears[i](x) for i,l in enumerate(self.linears)]

        return yhat

class Multisoftmax_ffnn_small(nn.Module):
    def __init__(self, input_size, output_vector):
        #super() gives access to methods in a
        #superclass from the subclass that inherits from it
        super(Multisoftmax_ffnn_small, self).__init__()
        self.lin1 = nn.Linear(input_size, 128) #
        self.bn1 = nn.BatchNorm1d(128)
        self.lin2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.lin3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)

        self.linears = nn.ModuleList([nn.Linear(32,i) for i in output_vector])
        #self.softmax = nn.ModuleList([nn.Softmax() for i in output_vector])

    def forward(self, x):
        x = F.relu(self.bn1(self.lin1(x)))
        x = F.relu(self.bn2(self.lin2(x)))
        x = F.relu(self.bn3(self.lin3(x)))

        yhat = [self.linears[i](x) for i,l in enumerate(self.linears)]

        return yhat

class Small_ffnn(nn.Module):
    def __init__(self, input_size, output_size):
        #super() gives access to methods in a
        #superclass from the subclass that inherits from it
        super(Small_ffnn, self).__init__()
        self.lin1 = nn.Linear(input_size, 512) #
        self.bn1 = nn.BatchNorm1d(512)
        self.lin2 = nn.Linear(512, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.lin3 = nn.Linear(128, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.head = nn.Linear(32, output_size)

    def forward(self, x):
        x = F.relu(self.bn1(self.lin1(x)))
        x = F.relu(self.bn2(self.lin2(x)))
        x = F.relu(self.bn3(self.lin3(x)))

        return torch.sigmoid(self.head(x))

class ffnn_1024x512x256x128(nn.Module):
    def __init__(self, input_size, output_size):
        #super() gives access to methods in a
        #superclass from the subclass that inherits from it
        super(ffnn_1024x512x256x128, self).__init__()
        self.lin1 = nn.Linear(input_size, 1024) #
        self.bn1 = nn.BatchNorm1d(1024)
        self.lin2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.lin3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.lin4 = nn.Linear(256, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.head = nn.Linear(128, output_size)

    def forward(self, x):
        x = F.relu(self.bn1(self.lin1(x)))
        x = F.relu(self.bn2(self.lin2(x)))
        x = F.relu(self.bn3(self.lin3(x)))
        x = F.relu(self.bn4(self.lin4(x)))

        return torch.sigmoid(self.head(x))



class Curiosity:
    def __init__(self, input_size,output_size, discrete_obs = None, train_every=500, mem_capacity=50000, batch_size=32):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        #self.curiosity_net = Small_ffnn(input_size,output_size).to(self.device)
        self.curiosity_net = ffnn_1024x512x256x128(input_size,output_size).to(self.device)
        self.curiosity_net.apply(init_weights)

        self.discrete_obs = discrete_obs

        self.memory_target = []
        self.memory_input = []
        self.train_counter = 1
        self.position = 0
        self.train_every = train_every
        self.mem_capacity = mem_capacity
        self.batch_size = batch_size

    def train(self, current_state, action, next_state, optimizer):

        current_state = self.sigmaEncoding(current_state)
        current_state = np.append(current_state,action)
        current_state = torch.from_numpy(current_state).float()

        #compute curiosity reward
        self.curiosity_net.eval()
        current_state = current_state.unsqueeze(0).to(self.device)
        pred_state=self.curiosity_net.forward(current_state)
        next_state = self.sigmaEncoding(next_state)
        next_state = torch.from_numpy(next_state).float().to(self.device)
        loss = F.mse_loss(input=pred_state,target=next_state)

        #save states into memory buffer
        #self.push(next_state, pred_state)
        self.push(next_state, current_state)

        if (self.train_counter % self.train_every) == 0 and (len(self.memory_target) >= self.batch_size):
            batch_mask = self.sample_index(self.batch_size)

            input_batch = [self.memory_input[i] for i in batch_mask]
            input_batch = torch.cat(input_batch)
            input_batch = torch.reshape(input_batch, (self.batch_size,-1))

            target_batch = [self.memory_target[i] for i in batch_mask]
            target_batch = torch.cat(target_batch)
            target_batch = torch.reshape(target_batch, (self.batch_size,-1))

            self.curiosity_net.train()
            prediction_batch=self.curiosity_net.forward(input_batch)

            loss_batch = F.mse_loss(input=prediction_batch,target=target_batch)
            optimizer.zero_grad()
            loss_batch.backward(retain_graph=True)
            optimizer.step()

        self.train_counter += 1

        return loss.cpu().detach().numpy()
        #return loss.detach().numpy()

    def push(self, next_state, current_state):
        #If we have space, we add a new slot
        if len(self.memory_target) < self.mem_capacity:
            self.memory_target.append(None)
            self.memory_input.append(None)
        self.memory_target[self.position] = next_state
        self.memory_input[self.position] = current_state
        #if we surpass the capacity, we restart the position to 0
        self.position = (self.position + 1) % self.mem_capacity

    def sample_index(self, batch_size):
        return random.sample(range(0,len(self.memory_target)), batch_size)

    def oneHotEncoding(self, observation):
        obs = np.array([])
        for i in range(len(self.discrete_obs)-1):
            zeros = np.zeros(self.discrete_obs[i])
            zeros[observation[i]] = 1
            obs = np.append(obs,zeros)
        return obs

    def sigmaEncoding(self, observation):

        sigmaEncoding = observation / self.discrete_obs
        #print(sigmaEncoding)
        #sys.exit()
        return sigmaEncoding
