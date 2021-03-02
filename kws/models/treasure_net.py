from torch import nn
from .attention import MultiHeadAttention
from .crnn import CRNN
import torch
#default
# num_keywords=1,
# time_steps=81,
# num_mels=40,
# conv_channels=16,
# kernel_size=(20, 5),
# stride=(8, 2),
# gru_hidden=256, gru_layers=2, num_heads=8, attention_layers=2, dropout=0.2

class TreasureNet(nn.Module):
    def __init__(self, num_keywords=1, time_steps=32, num_mels=64,
                 conv_channels=32, kernel_size=(8, 4), stride=(2, 2),
                 gru_hidden=256, gru_layers=2, num_heads=8, attention_layers=1, dropout=0.3):
        super(TreasureNet, self).__init__()
        self.encoder = CRNN(time_steps, num_mels, conv_channels, kernel_size, stride,
                            gru_hidden, gru_layers, dropout)

        self.layer_norm = nn.LayerNorm(gru_hidden)
        self.attention_layers = nn.ModuleList(MultiHeadAttention(gru_hidden, num_heads, dropout)
                                              for _ in range(attention_layers))

        self.classifier = nn.Linear(self.encoder.time_frames * gru_hidden, num_keywords + 1)

    def forward(self, inputs, hidden=None):
        inputs=torch.squeeze(inputs, 1)
        inputs = inputs.transpose(1, 2)
        # inputs: (batch_size, time_steps, num_mels)

        outputs, hidden = self.encoder(inputs.unsqueeze(1), hidden)
        # outputs: (batch_size, time_frames, gru_hidden)
        # hidden: (batch_size, gru_layers, gru_hidden)

        for attention in self.attention_layers:
            outputs = self.layer_norm(outputs)
            outputs = attention(query=outputs, key=outputs, value=outputs)
        # outputs: (batch_size, time_frames, gru_hidden)

        outputs = outputs.reshape(outputs.shape[0], -1)
        # outputs: (batch_size, time_frames * gru_hidden)

        outputs = self.classifier(outputs)
        # outputs: (batch_size, num_keywords + 1)

        return outputs #, hidden


def treasure_net(num_classes):
    # return TreasureNet(len(params['keywords']), params['time_steps'], params['num_mels'],
    #                    params['conv_channels'], params['kernel_size'], params['stride'],
    #                    params['gru_hidden'], params['gru_layers'], params['num_heads'],
    #                    params['attention_layers'], params['dropout'])
    return TreasureNet(num_keywords=num_classes-1)

