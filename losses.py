import torch
import torch.nn as nn

mse = nn.MSELoss(reduction="sum")

def final_states_loss(final_states, p_states):
    #cumulative xor
    cum_xor = 0.0
    for f in final_states:
        cum_xor = xor(cum_xor, p_states[f])
    return 1 - cum_xor

def not_final_states_loss(final_states, p_states):
    #cumulative and
    cum_and = 1.0
    for f in final_states:
        cum_and = conjunction(cum_and, negation(p_states[f]))
    return 1 - cum_and

def final_states_loss_log(final_states, p_states):
    #cumulative xor
    cum_xor = 0.0
    for f in final_states:
        cum_xor = xor(cum_xor, p_states[f])
    return torch.log(1 - cum_xor)

def final_states_loss_or(final_states, p_states):

    #cumulative xor
    cum_xor = 0.0
    for f in final_states:
        cum_xor = disjunction(cum_xor, p_states[f])
    return 1 - cum_xor

def entropy_loss(prob):
    prob = pi_0(prob)
    logprob = torch.log(prob)
    entrp_score = prob * logprob

    #sommare sulle classi
    entrp_score = torch.sum(entrp_score, dim=-1)

    return entrp_score.mean()

################################# projections
def pi_0(a, eps = 0.00001):
    return (1 - eps)*a + eps

def pi_1(a, eps = 0.00001):
    return (1 - eps)*a


################################# logical operations
# &&
def conjunction(a, b):
    return a*b

# ||
def disjunction(a,b):
    return a + b - a*b

# ->
def implication(pre,post) :
    return 1 - pre + pre*post

# !!
def negation(a):
    return 1 - a

def xor(a,b):
    # ( a or b ) and not ( a and b )
    return conjunction( disjunction(a,b) , negation( conjunction(a,b) )  )




