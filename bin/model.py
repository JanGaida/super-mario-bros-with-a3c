""" © Jan Gaida, 2020 """

# Torch
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch import cat as concatenate

# Für die initale Gewichte
import scipy.stats as stats


""" model.py

Definition des für das Programm zu grundeliegenden Neuralen-Netwerks


- ActorCriticModel:
Defintion des Actor-Critic-Netzwerkes

- DeepConvolutional:
Hilfsklasse für die Definition eines Convolutional-Netzwerk

- Conv2d_ReLU:
Hilfsklassse für eine nn.Conv2d mit ReLU-Aktivierungsfunktion

"""


class ActorCriticModel(nn.Module):
	"""Das zugrundeliegende Troch-Model für den A3C-Algorithmus"""

	def __init__(self, num_states, num_actions):
		super(ActorCriticModel, self).__init__()
		"""Init"""

		""" # CNN
		self.cnn = nn.Sequential(
			Conv2d_ReLU(num_states, 320, kernel_size=3, stride=2, padding=1),
			Conv2d_ReLU(320, 240, kernel_size=3, stride=2, padding=1),
			Conv2d_ReLU(240, 160, kernel_size=3, stride=2, padding=1),
			Conv2d_ReLU(160, 80, kernel_size=3, stride=2, padding=1)
		)
		"""

		# DeepConvolutional
		self.dcv = DeepConvolutional(num_states)

		# Memory
		memory_in_channels = 32*5*7*4 # Entspricht dem geflachten Output des vorherigen Netzes
		memory_out_channels = 512
		self.lstm = nn.LSTMCell(memory_in_channels, memory_out_channels) # LSTM-Version
		#self.gru = nn.GRUCell(memory_in_channels, memory_out_channels) # GRU-Version

		# Critc
		self.critic = nn.Linear(memory_out_channels, 1)

		# Actor
		self.actor = nn.Linear(memory_out_channels, num_actions)

		# Initialisiert die Gewichte
		self.init_weights()


	def init_weights(self):
		"""Hilfsfunktion um die initialen NN-Gewichte festzulegen"""
		
		for m in self.modules():
			# Actor / Critic
			if isinstance(m, nn.Linear):
				nn.init.xavier_uniform_(m.weight)
				nn.init.constant_(m.bias, 0)

			# LSTM / GRU
			elif isinstance(m, nn.LSTMCell) or isinstance(m, nn.GRUCell):
				nn.init.constant_(m.bias_ih, 0)
				nn.init.constant_(m.bias_hh, 0)

			# Convolutional
			elif isinstance(m, nn.Conv2d):
				#nn.init.xavier_uniform_(m.weight)
				#nn.init.constant_(m.bias, 0)
				values = T.as_tensor(stats.truncnorm(-2, 2, scale=0.01).rvs(m.weight.numel()), dtype=m.weight.dtype)
				with T.no_grad():
					m.weight.copy_(
						T.as_tensor( stats.truncnorm(-2, 2, scale=0.01).rvs(m.weight.numel()), dtype=m.weight.dtype).view(m.weight.size())
					)


	def forward(self, x, hx, cx): 
		"""Wenn das NN aufgerufen wird"""

		""" # CNN
		x = self.cnn(x)
		"""

		# DeepConvolutional
		x = self.dcv(x)

		# LSTM 
		hx, cx = self.lstm( x.view( x.size(0), -1), (hx,cx) )
		return self.actor(hx), self.critic(hx), hx, cx

		""" # GRU 
		hx = self.gru( x.view( x.size(0), -1), hx)
		return self.actor(hx), self.critic(hx), hx
		"""


class DeepConvolutional(nn.Module):
	"""Implementation eines Deep-Convolutional-Netzwerkes"""

	def __init__(self, in_channels):
		super(DeepConvolutional, self).__init__()
		"""Init"""

		# Branch A
		self.branch_a = nn.Sequential(
			Conv2d_ReLU(in_channels, 32, kernel_size=1),
			nn.AdaptiveAvgPool2d((5, 7))
		)
		# Branch B
		self.branch_b = nn.Sequential(
			Conv2d_ReLU(in_channels, 256, kernel_size=3, stride=2),
			Conv2d_ReLU(256, 96, kernel_size=3, stride=2),
			Conv2d_ReLU(96, 32, kernel_size=3, stride=2)
		)
		# Branch C
		self.branch_c = nn.Sequential(
			Conv2d_ReLU(in_channels, 256, kernel_size=5, stride=3, padding=1),
			Conv2d_ReLU(256, 32, kernel_size=5, stride=3, padding=1)
		)
		# Branch D
		self.branch_d = nn.Sequential(
			nn.AdaptiveMaxPool2d((5, 7)),
			Conv2d_ReLU(in_channels, 32, kernel_size=1)
		)
		# Pool
		self.pool = nn.AvgPool2d(1)

	def forward(self, x):
		"""Wenn das DeepConvolutional aufgerufen wird"""
		return self.pool(concatenate([self.branch_a(x), self.branch_b(x), self.branch_c(x), self.branch_d(x)], dim=1))


class Conv2d_ReLU(nn.Module):
	"""Hilfsklasse für die Implementation eines Conv2d mit ReLU-Aktivierungsfunktion"""

	def __init__(self, in_channels, out_channels, **kwargs):
		super(Conv2d_ReLU, self).__init__()
		"""Init"""
		# Conv2d mit den übergebenen Parametern
		self.cv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)

	def forward(self, x):
		"""Wenn die Conv2d_ReLU aufgerufen wird"""
		return F.relu(self.cv(x))
