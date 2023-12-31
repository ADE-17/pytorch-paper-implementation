
import torch
import torch.nn as nn
from model import Conformer

batch_size, sequence_length, dim = 3, 12345, 80
cuda = torch.cuda.is_available()  
device = torch.device('cuda' if cuda else 'cpu')

inputs = torch.rand(batch_size, sequence_length, dim).to(device)
input_lengths = torch.LongTensor([12345, 12300, 12000])
targets = torch.LongTensor([[1, 3, 3, 3, 3, 3, 4, 5, 6, 2],
                            [1, 3, 3, 3, 3, 3, 4, 5, 2, 0],
                            [1, 3, 3, 3, 3, 3, 4, 2, 0, 0]]).to(device)
target_lengths = torch.LongTensor([9, 8, 7])

model = Conformer(num_class=10,
                  input_dim=dim,
                  encoder_dim=32,
                  num_encoder_layer=3).to(device)

criterion = nn.CTCLoss().to(device)
optimizer = torch.optim.SGD(params=model.parameters())
outputs, output_lengths = model(inputs, input_lengths)
loss = criterion(outputs.transpose(0, 1), targets, output_lengths, target_lengths)
print (loss)
