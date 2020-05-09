# Torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCriticModel(nn.Module):
    """Das zugrundeliegende Troch-Model für den A3C-Algorithmus"""

    def __init__(self, num_states, num_actions):
        """Init"""
        super(ActorCriticModel, self).__init__()

        # CNN
        self.conv1 = nn.Conv2d(num_states, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

        # LSTM
        self.lstm = nn.LSTMCell(32 * 6 * 6, 512)

        # Critc
        self.critic = nn.Linear(512, 1)

        # Actor
        self.actor = nn.Linear(512, num_actions)

        # Initialisiert die Gewichte
        self.init_weights()


    def init_weights(self):
        """Hilfsfunktion um die initialen NN-Gewichte festzulegen"""
        for m in self.modules():

            # CNN
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

            # Actor / Critic
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

            # LSTM
            elif isinstance(m, nn.LSTMCell):
                nn.init.constant_(m.bias_ih, 0)
                nn.init.constant_(m.bias_hh, 0)


    def forward(self, x, hx, cx):
        """Wenn das NN aufgreufen wird"""

        # CNN
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        # LSTM
        hx, cx = self.lstm(x.view(x.size(0), -1), (hx, cx))

        return self.actor(hx), self.critic(hx), hx, cx


