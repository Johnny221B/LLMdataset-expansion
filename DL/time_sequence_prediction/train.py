from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
import time
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class Sequence(nn.Module):
    def __init__(self):
        super(Sequence, self).__init__()
        self.lstm1 = nn.LSTMCell(1, 51)
        self.lstm2 = nn.LSTMCell(51, 51)
        self.linear = nn.Linear(51, 1)

    def forward(self, input, future=0):
        outputs = []
        h_t = torch.zeros(input.size(0), 51, dtype=torch.double, device=input.device)
        c_t = torch.zeros(input.size(0), 51, dtype=torch.double, device=input.device)
        h_t2 = torch.zeros(input.size(0), 51, dtype=torch.double, device=input.device)
        c_t2 = torch.zeros(input.size(0), 51, dtype=torch.double, device=input.device)

        for input_t in input.split(1, dim=1):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        for i in range(future):
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        outputs = torch.cat(outputs, dim=1)
        return outputs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=15, help='steps to run')
    opt = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    np.random.seed(0)
    torch.manual_seed(0)

    data = torch.load('/root/autodl-tmp/netthink/DL/time_sequence_prediction/traindata.pt')
    input = torch.from_numpy(data[3:, :-1]).double().to(device)
    target = torch.from_numpy(data[3:, 1:]).double().to(device)
    test_input = torch.from_numpy(data[:3, :-1]).double().to(device)
    test_target = torch.from_numpy(data[:3, 1:]).double().to(device)

    seq = Sequence().double().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.LBFGS(seq.parameters(), lr=0.8)

    start_time = time.time()

    for i in range(opt.steps):
        print('STEP: ', i)
        def closure():
            optimizer.zero_grad()
            out = seq(input)
            loss = criterion(out, target)
            print('loss:', loss.item())
            loss.backward()
            return loss
        optimizer.step(closure)

        with torch.no_grad():
            future = 1000
            pred = seq(test_input, future=future)
            loss = criterion(pred[:, :-future], test_target)
            print('test loss:', loss.item())
            y = pred.cpu().detach().numpy()

        plt.figure(figsize=(30,10))
        plt.title('Predict future values for time sequences\n(Dashlines are predicted values)', fontsize=30)
        plt.xlabel('x', fontsize=20)
        plt.ylabel('y', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        def draw(yi, color):
            plt.plot(np.arange(input.size(1)), yi[:input.size(1)], color, linewidth=2.0)
            plt.plot(np.arange(input.size(1), input.size(1) + future), yi[input.size(1):], color + ':', linewidth=2.0)
        draw(y[0], 'r')
        draw(y[1], 'g')
        draw(y[2], 'b')
        plt.savefig('predict%d.pdf' % i)
        plt.close()

    elapsed_time = time.time() - start_time
    print('Total training and prediction time: {:.2f} seconds'.format(elapsed_time))
