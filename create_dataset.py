import random
from itertools import product
import torch

########################################################################################################################################
def create_complete_set_traces(max_length_traces, alphabet, dfa, train_size,train_with_accepted_only, verbose=False):
    traces = []
    traces_t = []
    accepted = []

    for length_traces in range(1, max_length_traces+1):
        #AD HOC FOR 2 SYMBOLS
        prod = product([0,1,2,3], repeat=length_traces)

        for trace in list(prod):
            t = []
            t_t = torch.zeros((len(trace), len(alphabet)))

            for step, true_literal in enumerate(trace):
                truth_v = {}
                if true_literal % 2 == 0:
                    truth_v[alphabet[0]] = True
                    t_t[step, 0] = 1.0
                else:
                    truth_v[alphabet[0]] = False

                if true_literal < 2:
                    truth_v[alphabet[1]] = True
                    t_t[step, 1] = 1.0
                else:
                    truth_v[alphabet[1]] = False

                t.append(truth_v)

            traces.append(t)
            traces_t.append(t_t)
            if dfa.accepts(t):
                accepted.append(1)
            else:
                accepted.append(0)

    #shuffle
    dataset = list(zip(traces, traces_t, accepted))
    random.shuffle(dataset)
    traces, traces_t, accepted = zip(*dataset)

    if verbose:
        print("----TRACES:----")
        for i in range(len(traces)):
            print(traces[i])
            print(traces_t[i])
            if accepted[i] == 1:
                print("YES")
            else:
                print("NO")
        print("------------------------")

    #split
    split_index = round(len(traces) * train_size)

    if not train_with_accepted_only:
        traces_train = traces[:split_index]
        traces_test = traces[split_index:]

        traces_t_train = traces_t[:split_index]
        traces_t_test = traces_t[split_index:]

        accepted_train = accepted[:split_index]
        accepted_test = accepted[split_index:]
    else:
        traces_train = []
        traces_test = []
        traces_t_train = []
        traces_t_test = []
        accepted_train = []
        accepted_test = []

        index = 0
        for i in range(len(traces)):
            if index < split_index and accepted[i] == 1:
                traces_train.append(traces[i])
                traces_t_train.append(traces_t[i])
                accepted_train.append(accepted[i])
            else:
                traces_test.append(traces[i])
                traces_t_test.append(traces_t[i])
                accepted_test.append(accepted[i])


    print(
        "created symbolic dataset with all the {} traces of maximum length {}; {} train, {} test".format(len(traces), max_length_traces, len(traces_train), len(traces_test)))

    return traces_train, traces_test, traces_t_train, traces_t_test, accepted_train, accepted_test


# return:    X = list of traces
#           y = list of values of acceptance (0 or 1)
def create_complete_set_traces_one_true_literal(max_length_traces, alphabet, dfa, train_size,train_with_accepted_only, verbose=False): #<----------------------------------
    traces = []
    traces_t = []
    accepted = []

    for length_traces in range(1, max_length_traces+1):
        prod = product(alphabet, repeat=length_traces)

        for trace in list(prod):
            t = []
            t_t = torch.zeros((len(trace), len(alphabet)))

            for step, true_literal in enumerate(trace):
                truth_v = {}
                for s, symbol in enumerate(alphabet):
                    if symbol == true_literal:
                        truth_v[symbol] = True
                        t_t[step, s] = 1.0
                    else:
                        truth_v[symbol] = False

                t.append(truth_v)
            traces.append(t)
            traces_t.append(t_t)
            if dfa.accepts(t):
                accepted.append(1)
            else:
                accepted.append(0)

    #shuffle
    dataset = list(zip(traces, traces_t, accepted))
    random.shuffle(dataset)
    traces, traces_t, accepted = zip(*dataset)

    if verbose:
        print("----TRACES:----")
        for i in range(len(traces)):
            print(traces[i])
            print(traces_t[i])
            if accepted[i] == 1:
                print("YES")
            else:
                print("NO")
        print("------------------------")

    #split
    split_index = round(len(traces) * train_size)

    if not train_with_accepted_only:
        traces_train = traces[:split_index]
        traces_test = traces[split_index:]

        traces_t_train = traces_t[:split_index]
        traces_t_test = traces_t[split_index:]

        accepted_train = accepted[:split_index]
        accepted_test = accepted[split_index:]
    else:
        traces_train = []
        traces_test = []
        traces_t_train = []
        traces_t_test = []
        accepted_train = []
        accepted_test = []

        index = 0
        for i in range(len(traces)):
            if index < split_index and accepted[i] == 1:
                traces_train.append(traces[i])
                traces_t_train.append(traces_t[i])
                accepted_train.append(accepted[i])
            else:
                traces_test.append(traces[i])
                traces_t_test.append(traces_t[i])
                accepted_test.append(accepted[i])


    print("created symbolic dataset with all the {} traces of maximum length {}; {} train, {} test".format(len(traces), max_length_traces, len(traces_train), len(traces_test)))

    return traces_train, traces_test, traces_t_train, traces_t_test, accepted_train, accepted_test

def create_image_sequence_dataset_non_mut_ex(image_data, numb_of_classes, traces, acceptance):
    channels = 1
    pixels_h, pixels_v = image_data.data[0].size()
    how_many = []
    data_for_classes = []
    for label in range(numb_of_classes):
        indices_i = image_data.targets == label
        data_i, target_i = image_data.data[indices_i], image_data.targets[indices_i]
        how_many.append(len(data_i))
        data_for_classes.append(data_i)

    num_of_images = sum(how_many)

    img_seq_train = []
    acceptance_train = []


    i_i = [0 for _ in range(len(how_many)) ]
    seen_images = sum(i_i)


    while True:
        for j in range(len(traces)):
            x = traces[j]
            a = acceptance[j]
            num_img = len(x)
            x_i_img = torch.zeros(num_img, channels,pixels_h, pixels_v)

            for step in range(num_img):
                if x[step][0] > 0.5:

                    x_i_img[step] += data_for_classes[0][i_i[0]]
                    i_i[0] += 1
                    if i_i[0] >= how_many[0]:
                        break
                if x[step][1] > 0.5:
                    x_i_img[step] += data_for_classes[1][i_i[1]]
                    i_i[1] += 1
                    if i_i[1] >= how_many[1]:
                        break
            if i_i[0] >= how_many[0] or i_i[1] >= how_many[1]:
                break
            img_seq_train.append(x_i_img)
            acceptance_train.append(a)

            seen_images +=num_img
        if i_i[0] >= how_many[0] or i_i[1] >= how_many[1]:
            break

    #print("Created image dataset with {} sequences: {} train, {} test".format(len(img_seq_train) + len(img_seq_test), len(img_seq_train), len(img_seq_test)))

    return img_seq_train, acceptance_train

def create_image_sequence_dataset(image_data, numb_of_classes, traces, acceptance):
    channels = 1
    pixels_h, pixels_v = image_data.data[0].size()
    how_many = []
    data_for_classes = []
    for label in range(numb_of_classes):
        indices_i = image_data.targets == label
        data_i, target_i = image_data.data[indices_i], image_data.targets[indices_i]
        how_many.append(len(data_i))
        data_for_classes.append(data_i)

    num_of_images = sum(how_many)

    img_seq_train = []
    acceptance_train = []
    img_seq_test = []
    acceptance_test = []

    i_i = [0 for _ in range(len(how_many)) ]
    seen_images = sum(i_i)


    while True:
        for j in range(len(traces)):
            x = traces[j]
            a = acceptance[j]
            num_img = len(x)
            x_i_img = torch.zeros(num_img, channels,pixels_h, pixels_v)

            for step in range(num_img):
                if x[step][0] > 0.5:

                    x_i_img[step] = data_for_classes[0][i_i[0]]
                    i_i[0] += 1
                    if i_i[0] >= how_many[0]:
                        break
                else:
                    x_i_img[step] = data_for_classes[1][i_i[1]]
                    i_i[1] += 1
                    if i_i[1] >= how_many[1]:
                        break
            if i_i[0] >= how_many[0] or i_i[1] >= how_many[1]:
                break
            img_seq_train.append(x_i_img)
            acceptance_train.append(a)

            seen_images +=num_img
        if i_i[0] >= how_many[0] or i_i[1] >= how_many[1]:
            break

    #print("Created image dataset with {} sequences: {} train, {} test".format(len(img_seq_train) + len(img_seq_test), len(img_seq_train), len(img_seq_test)))

    return img_seq_train, acceptance_train
#############################################################################################################################################################
