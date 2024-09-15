import torch
from torch import nn
from torch.nn import functional as F

# Sample from the Gumbel-Softmax distribution and optionally discretize.
class GumbelSoftmax(nn.Module):

    def __init__(self, f_dim, c_dim):
        super(GumbelSoftmax, self).__init__()
        self.logits = nn.Linear(f_dim, c_dim)
        self.f_dim = f_dim
        self.c_dim = c_dim
        
    def sample_gumbel(self, shape, is_cuda=False, eps=1e-20):
        U = torch.rand(shape)
        if is_cuda:
            U = U.cuda()
        return -torch.log(-torch.log(U + eps) + eps)

    def gumbel_softmax_sample(self, logits, temperature):
        y = logits + self.sample_gumbel(logits.size(), logits.is_cuda)
        return F.softmax(y / temperature, dim=-1)

    def gumbel_softmax(self, logits, temperature, hard=False):
        """
        ST-gumple-softmax
        input: [*, n_class]
        return: flatten --> [*, n_class] an one-hot vector
        """
        #categorical_dim = 10
        y = self.gumbel_softmax_sample(logits, temperature)

        if not hard:
            return y

        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        # Set gradients w.r.t. y_hard gradients w.r.t. y
        y_hard = (y_hard - y).detach() + y
        return y_hard 

    def forward(self, x, temperature=1.0, hard=False):
        logits = self.logits(x).view(-1, self.c_dim)
        prob = F.softmax(logits, dim=-1)
        y = self.gumbel_softmax(logits, temperature, hard)
        return logits, prob, y
    

# Sample from the Gumbel-Softmax distribution and optionally discretize.
class GumbelSoftmax_3D(nn.Module):

    def __init__(self, f_dim, d_dim, c_dim):
        super(GumbelSoftmax_3D, self).__init__()
        self.logits = nn.Linear(f_dim, c_dim * d_dim)
        self.f_dim = f_dim
        self.c_dim = c_dim
        self.d_dim = d_dim

        nn.init.xavier_uniform_(self.logits.weight)
        
    def sample_gumbel(self, shape, is_cuda=False, eps=1e-20):
        U = torch.rand(shape)
        if is_cuda:
            U = U.cuda()
        return -torch.log(-torch.log(U + eps) + eps)

    def gumbel_softmax_sample(self, logits, temperature):
        y = logits + self.sample_gumbel(logits.size(), logits.is_cuda)
        return F.softmax(y / temperature, dim=-1)

    def gumbel_softmax(self, logits, temperature, hard=False):
        """
        ST-gumple-softmax
        input: [*, n_class]
        return: flatten --> [*, n_class] an one-hot vector
        """
        #categorical_dim = 10
        y = self.gumbel_softmax_sample(logits, temperature)

        if not hard:
            return y

        shape = y.size()
        _, ind = y.max(dim=-1)
        # y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard = torch.zeros_like(y)
        # y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard.scatter_(2, ind.unsqueeze(-1), 1)
        y_hard = y_hard.view(*shape)
        # Set gradients w.r.t. y_hard gradients w.r.t. y
        y_hard = (y_hard - y).detach() + y
        return y_hard 

    def forward(self, x, temperature=1.0, hard=False):
        logits = self.logits(x).view(-1, self.d_dim, self.c_dim)
        prob = F.softmax(logits, dim=-1)
        y = self.gumbel_softmax(logits, temperature, hard)
        return logits, prob, y
    

class LatentODE(nn.Module):
    """
    A class modelling the latent splicing dynamics.

    Parameters
    ----------
    n_latent
        Dimension of latent space.
        (Default: 20)
    n_hidden
        The dimensionality of the hidden layer for the ODE function.
        (Default: 128)
    """

    def __init__(
        self,
        n_latent: int = 20,
        n_hidden: int = 128,
    ):
        super().__init__()
        self.n_latent = n_latent
        # self.elu = nn.ELU()
        self.fc1 = nn.Linear(n_latent, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, n_latent)

    def forward(self, t: torch.Tensor, x: torch.Tensor):
        """
        Compute the gradient at a given time t and a given state x.

        Parameters
        ----------
        t
            A given time point.
        x
            A given spliced latent state.

        Returns
        ----------
        :class:`torch.Tensor`
            A tensor
        """
        x_in = torch.log(1e-8 + torch.cat((x[..., :int(self.n_latent / 2)], \
                                           torch.sqrt(x[..., int(self.n_latent / 2):] + 1e-10)), \
                                           dim=-1))
        out = self.fc1(x_in)
        out = F.elu(out)
        out = self.fc2(out)
        out = F.elu(out)
        out = F.softplus(self.fc3(out))
        return out