# Torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCriticModel(nn.Module):
    """Das zugrundeliegende Troch-Model für den A3C-Algorithmus"""

    def __init__(self, num_states, num_actions):
        super(ActorCriticModel, self).__init__()
        """Init"""

        # CNN
        self.conv1 = nn.Conv2d(num_states, 640, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(640, 320, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(320, 160, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(160, 80, 3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(80, 40, 3, stride=2, padding=1)
        """ Ganz gut
        self.conv1 = nn.Conv2d(num_states, 320, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(320, 160, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(160, 80, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(80, 40, 3, stride=2, padding=1)
        """
        """ Besser
        self.conv1 = nn.Conv2d(num_states, 320, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(320, 240, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(240, 160, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(160, 80, 3, stride=2, padding=1)
        """

        # Memory
        self.lstm = nn.LSTMCell(40 * 2 * 2, 512) # LSTM-Version
        #self.gru = nn.GRUCell(80 * 4 * 4, 512) # GRU-Version

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

            # LSTM / GRU
            elif isinstance(m, nn.LSTMCell) or isinstance(m, nn.GRUCell):
                nn.init.constant_(m.bias_ih, 0)
                nn.init.constant_(m.bias_hh, 0)

    # LSTM-Version
    def forward(self, x, hx, cx): 
        # Wenn das NN aufgerufen wird

        # CNN
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))

        #print("~> CNN-OUTPUT-SIZE: {}".format(x.size()))

        hx, cx = self.lstm( x.view( x.size(0), -1), (hx,cx))
        return self.actor(hx), self.critic(hx), hx, cx
    """
    # GRU-Version
    def forward(self, x, hx):
        # Wenn das NN aufgerufen wird

        # CNN
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        #print("~> CNN-OUTPUT-SIZE: {}".format(x.size()))

        hx = self.gru( x.view( x.size(0), -1), hx)
        return self.actor(hx), self.critic(hx), hx
    """
