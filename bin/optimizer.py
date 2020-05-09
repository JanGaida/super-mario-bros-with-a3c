import torch as T

class Adam(T.optim.Adam):
    """Der Adam-Optimierer"""

    def __init__(self, params, args):
        """Init"""
        super(Adam, self).__init__(params, lr=args.learning_rate)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]

                # Initale Werte
                state['step'] = 0
                state['exp_avg'] = T.zeros_like(p.data)
                state['exp_avg_sq'] = T.zeros_like(p.data)

                # Für alle Modele
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()
