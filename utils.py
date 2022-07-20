import torch
import random
from numpy.random import RandomState
import os
import numpy as np
if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

def set_seed(seed: int) -> RandomState:
    """ Method to set seed across runs to ensure reproducibility.
    It fixes seed for single-gpu machines.
    Args:
        seed (int): Seed to fix reproducibility. It should different for
            each run
    Returns:
        RandomState: fixed random state to initialize dataset iterators
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False # set to false for reproducibility, True to boost performance
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    random_state = random.getstate()
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    return random_state

def eval_acceptance(classifier, automa, final_states, dfa, alphabet, dataset, automa_implementation='dfa', mutually_exc_sym=True):
    #automa implementation =
    #   - 'dfa' use the perfect dfa given
    #   - 'lstm' use the lstm model
    #   - 'logic_circuit' use the fuzzy automaton
    total = 0
    correct = 0
    test_loss = 0
    classifier.eval()


    with torch.no_grad():
        for i in range(len(dataset[0])):
            images = dataset[0][i].to(device)
            label = dataset[1][i]

            # primo modo usando la lstm o l'automa continuo
            if automa_implementation == 'lstm':
                accepted = automa(classifier(images))
                accepted = accepted[-1]

                output = torch.argmax(accepted).item()


            #secondo modo usando l'automa
            elif automa_implementation == 'dfa':
                pred_labels = classifier(images)
                if mutually_exc_sym:
                    pred_labels = pred_labels.data.max(1, keepdim=False)[1]

                    trace = []
                    for p_l in pred_labels:
                        truth_v = {}
                        for symbol in alphabet:
                            truth_v[symbol] = False

                        truth_v[alphabet[p_l.item()]] = True
                        trace.append(truth_v)
                else:
                    trace = []

                    for pred in pred_labels:
                        truth_v = {}
                        for i, symbol in enumerate(alphabet):
                            if pred[i] > 0.5:
                                truth_v[symbol] = True
                            else:
                                truth_v[symbol] = False
                        trace.append(truth_v)

                output = int(dfa.accepts(trace))

            #terzo modo: usando il circuito logico continuo
            elif automa_implementation == 'logic_circuit':
                sym = classifier(images)

                last_state = automa(sym)
                last_state = torch.argmax(last_state).item()

                output = int(last_state in final_states)

            else:
                print("INVALID AUTOMA IMPLEMENTATION: ", automa_implementation)

            total += 1


            correct += int(output==label)

            test_accuracy = 100. * correct/(float)(total)

    return test_accuracy

def eval_image_classification_from_traces(traces_images, traces_labels, classifier, mutually_exclusive):
    total = 0
    correct = 0
    classifier.eval()


    with torch.no_grad():
        for i in range(len(traces_labels)) :
            t_sym = traces_labels[i].to(device)
            t_img = traces_images[i].to(device)

            pred_sym = classifier(t_img)

            if  not mutually_exclusive:

                y1 = torch.ones(t_sym.size())
                y2 = torch.zeros(t_sym.size())

                output_sym = pred_sym.where(pred_sym <= 0.5, y1)
                output_sym = output_sym.where(pred_sym > 0.5, y2)

                correct += torch.sum(output_sym == t_sym).item()
                total += torch.numel(pred_sym)

            else:
                output_sym = pred_sym.data.max(1, keepdim=True)[1]

                t_sym = t_sym.data.max(1, keepdim=True)[1]

                correct += torch.sum(output_sym == t_sym).item()
                total += t_sym.size()[0]

    accuracy = 100. * correct / (float)(total)
    return accuracy

