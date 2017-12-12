import torch
from functools import partial

C = partial(torch.autograd.Variable, requires_grad=False)
V = partial(torch.autograd.Variable, requires_grad=True)

# alias tensor sources. makes life easier
tensors = torch.cuda if torch.cuda.is_available() else torch
