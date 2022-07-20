import torch

import torch.nn as nn
import torch.nn.functional as F
from losses import conjunction
from losses import disjunction, negation

if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

class LSTMAutoma(nn.Module):

    def __init__(self, hidden_dim, vocab_size, tagset_size):
        super(LSTMAutoma, self).__init__()
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(vocab_size, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):

        lstm_out, _ = self.lstm(sentence.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))

        return tag_space

    def predict(self, sentence):
        tag_space = self.forward(sentence)
        out = F.softmax(tag_space, dim=1)[-1]
        return out

class FuzzyAutoma(nn.Module):
    def __init__(self, numb_of_symbols, numb_of_states, reduced_dfa):
        super(FuzzyAutoma, self).__init__()
        self.dfa = reduced_dfa
        self.numb_of_symbols = numb_of_symbols
        self.numb_of_states = numb_of_states

    #input: sequence of symbols probabilities (N, num_of_symbols)
    def forward(self, symbols_prob):
        s = torch.zeros(self.numb_of_states)
        #initial state is 0 for construction
        s[0] = 1.0

        for action in symbols_prob:
            s = self.next_state(s, action)

        return s

    def next_state(self,state, action):
        nxt_stt = torch.zeros(state.size()).to(device)
        for s in self.dfa.keys():
            for sym in self.dfa[s].keys():
                nxt_stt[self.dfa[s][sym]] += conjunction(state[s], action[sym])
                #print("conj fra state={} and sym={} è {} e viene scritta in {}".format(s, sym, c, self.dfa[s][sym]))
        #return F.softmax(nxt_stt, dim=0)
        return nxt_stt

class FuzzyAutoma_non_mutex(nn.Module):
    def __init__(self, numb_of_symbols, numb_of_states, reduced_dfa):
        super(FuzzyAutoma, self).__init__()
        self.dfa = reduced_dfa
        self.numb_of_symbols = numb_of_symbols
        self.numb_of_states = numb_of_states

    #input: sequence of symbols probabilities (N, num_of_symbols)
    def forward(self, symbols_prob):
        s = torch.zeros(self.numb_of_states)
        #initial state is 0 for construction
        s[0] = 1.0

        for action in symbols_prob:
            #print(s)
            s = self.next_state(s, action)

        return s

    def next_state(self,state, action):
        nxt_stt = torch.zeros(state.size()).to(device)
        for s in self.dfa.keys():
            for sym in self.dfa[s].keys():
                action_guard = recursive_guard_evaluation(sym, action)
                #print(action_guard)
                #print(self.dfa[s][sym])
                vvv = conjunction(state[s], action_guard)
                #print(vvv)

                nxt_stt[self.dfa[s][sym]] += vvv
                #print("conj fra state={} and sym={} è {} è {} e viene scritta in {}".format(s, sym, action_guard, vvv, self.dfa[s][sym]))
                #print("nxt state : ", nxt_stt)
                #print("sftmx of nxt state : ",F.softmax(nxt_stt, dim=0))
        #return F.softmax(nxt_stt, dim=0)
        return nxt_stt

#input : String
#return: [argument1 , argument 2, .. , argument n ]
def divide_args_n(guard):
    #print("guard: ", guard)
    args = guard.split(',')
    done = False
    args_str = []

    curr_arg_i = 0
    while curr_arg_i < len(args):
        arg0 = args[curr_arg_i]
        while arg0.count('(') != arg0.count(')'):
            curr_arg_i += 1
            arg0 = arg0 + ',' + args[curr_arg_i]
        args_str.append(arg0)
        curr_arg_i += 1
    return args_str


#guard : String
#action_prob: Tensor
def recursive_guard_evaluation(guard, action):
    #print("guard: ", guard)
    #and
    if guard[0] == 'a':

        guard = guard[4:-1]
        value = 1.0
        args = divide_args_n(guard)
        for arg in args:
            value = conjunction(value, recursive_guard_evaluation(arg, action))
        return value
    #or
    elif guard[0] == 'o':

        guard = guard[3:-1]
        args = divide_args_n(guard)
        value = 0.0
        for arg in args:
            value = disjunction(value, recursive_guard_evaluation(arg, action))
        return value

    #not
    elif guard[0] == 'n':
        guard = guard[4:-1]
        return negation(recursive_guard_evaluation(guard, action))

    #true
    elif guard[0] == 'T':
        return 1.0

    #pure symbol
    else:
        sym = int(guard)
        return action[sym]


#guard = sympy formula
#output= string
def recurrent_write_guard( guard):
        if str(type(guard)) == "And":
            args = list(guard._argset)
            string_g = "and("
            for arg in args:
                string_g = string_g + recurrent_write_guard(arg) + ","
            string_g = string_g[:-1]
            string_g += ")"
            return string_g
        if str(type(guard)) == "Or":
            args = list(guard._argset)
            string_g = "or("
            for arg in args:
                string_g = string_g + recurrent_write_guard(arg) + ","
            string_g = string_g[:-1]
            string_g += ")"
            return string_g
        if str(type(guard)) == "Not":
            #Nota: non funziona se metto in not fomule più lunghe del singolo simbolo
            arg = str(guard)[2:]

            return "not({})".format(arg)
        if str(type(guard)) == "<class 'sympy.core.symbol.Symbol'>":
            #print(str(guard))
            #print(str(guard)[1:])
            return str(guard)[1:]
        if str(type(guard)) == "<class 'sympy.logic.boolalg.BooleanTrue'>":
            return 'T'
        else:
            print("Not recognized type for the guard: ", type(guard))
            assert(3==0)