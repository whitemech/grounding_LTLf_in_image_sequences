import torch

from DeepAutoma import LSTMAutoma, FuzzyAutoma, FuzzyAutoma_non_mutex, recurrent_write_guard
from Classifier import CNN
from losses import final_states_loss, not_final_states_loss
import itertools
import math
from utils import eval_acceptance, eval_image_classification_from_traces
if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

class LTL_grounding:
    def __init__(self, ltl_formula, dfa, mutex, symbolic_dataset, image_seq_dataset, C, N, T, dataset='MNIST', train_with_accepted_only= True, automa_implementation = 'logic_circuit', lstm_output= "acceptance", num_exp=0,log_dir="Results/"):
        self.log_dir = log_dir
        self.exp_num=num_exp
        self.ltl_formula_string = ltl_formula
        self.dfa = dfa
        self.mutually_exclusive = mutex
        #save the dfa image
        self.dfa.to_graphviz().render("Automas/"+self.ltl_formula_string)

        self.numb_of_symbols = C
        self.length_traces = T
        self.numb_of_states = self.dfa._state_counter

        self.alphabet = ["c"+str(i) for i in range(C) ]
        self.final_states = list(self.dfa._final_states)

        #reduced dfa for single label image classification
        if self.mutually_exclusive:
            self.reduced_dfa = self.reduce_dfa()
        else:
            self.reduced_dfa = self.reduce_dfa_non_mutex()

        print("DFA: ",self.dfa._transition_function)

        #################### networks
        self.hidden_dim =6
        self.automa_implementation = automa_implementation


        if self.automa_implementation == 'lstm':
            if lstm_output== "states":
                self.deepAutoma = LSTMAutoma(self.hidden_dim, self.numb_of_symbols, self.numb_of_states)
            elif lstm_output == "acceptance":
                self.deepAutoma = LSTMAutoma(self.hidden_dim, self.numb_of_symbols, 2)
            else:
                print("INVALID LSTM OUTPUT. Choose between 'states' and 'acceptance'")
        elif self.automa_implementation == 'logic_circuit':
            if self.mutually_exclusive:
                self.deepAutoma = FuzzyAutoma(self.numb_of_symbols, self.numb_of_states, self.reduced_dfa)
            else:
                self.deepAutoma = FuzzyAutoma_non_mutex(self.numb_of_symbols, self.numb_of_states, self.reduced_dfa)
        else:
            print("INVALID AUTOMA IMPLEMENTATION. Choose between 'lstm' and 'logic_circuit'")

        if dataset == 'MNIST':
            self.num_classes = 2
            self.num_channels = 1
            nodes_linear = 54

            self.pixels_h = 28
            self.pixels_v = 28
            self.num_features = 4

        self.classifier = CNN(self.num_channels, self.num_classes, nodes_linear, self.mutually_exclusive)

        #dataset
        self.train_traces, self.test_traces, train_acceptance_tr, test_acceptance_tr = symbolic_dataset
        self.train_img_seq, self.train_acceptance_img, self.test_img_seq_clss, self.test_acceptance_img_clss, self.test_img_seq_aut, self.test_acceptance_img_aut, self.test_img_seq_hard, self.test_acceptance_img_hard = image_seq_dataset

    def reduce_dfa(self):
        dfa = self.dfa

        admissible_transitions = []
        for true_sym in self.alphabet:
            trans = {}
            for i,sym in enumerate(self.alphabet):
                trans[sym] = False
            trans[true_sym] = True
            admissible_transitions.append(trans)
        red_trans_funct = {}
        for s0 in self.dfa._states:
            red_trans_funct[s0] = {}
            transitions_from_s0 = self.dfa._transition_function[s0]
            for key in transitions_from_s0:
                label = transitions_from_s0[key]
                for sym, at in enumerate(admissible_transitions):
                    if label.subs(at):
                        red_trans_funct[s0][sym] = key

        return red_trans_funct

    def reduce_dfa_non_mutex(self):

        red_trans_funct = {}
        for s0 in self.dfa._states:
            #print("s0 :", s0)
            red_trans_funct[s0] = {}
            transitions_from_s0 = self.dfa._transition_function[s0]
            for key in transitions_from_s0:
                label = transitions_from_s0[key]
                #print("guard: ", label)
                label = recurrent_write_guard(label)
                #print("guard string : ", label)

                red_trans_funct[s0][label] = key

        return red_trans_funct

    def eval_automa_acceptance(self, automa_implementation):
        train_accuracy = eval_acceptance(self.classifier, self.deepAutoma, self.final_states, self.dfa, self.alphabet,(self.train_img_seq, self.train_acceptance_img), automa_implementation, mutually_exc_sym=self.mutually_exclusive)
        test_accuracy_clss= eval_acceptance( self.classifier, self.deepAutoma, self.final_states, self.dfa, self.alphabet,(self.test_img_seq_clss, self.test_acceptance_img_clss), automa_implementation, mutually_exc_sym=self.mutually_exclusive)
        test_accuracy_aut= eval_acceptance( self.classifier, self.deepAutoma, self.final_states, self.dfa, self.alphabet,(self.test_img_seq_aut, self.test_acceptance_img_aut), automa_implementation, mutually_exc_sym=self.mutually_exclusive)
        test_accuracy_hard= eval_acceptance( self.classifier, self.deepAutoma, self.final_states, self.dfa,self.alphabet,(self.test_img_seq_hard, self.test_acceptance_img_hard), automa_implementation, mutually_exc_sym=self.mutually_exclusive)

        return train_accuracy, test_accuracy_clss, test_accuracy_aut, test_accuracy_hard

    def eval_image_classification(self):
        train_acc = eval_image_classification_from_traces(self.train_img_seq, self.train_traces, self.classifier, self.mutually_exclusive)
        test_acc = eval_image_classification_from_traces(self.test_img_seq_hard, self.test_traces, self.classifier, self.mutually_exclusive)
        return train_acc, test_acc


    def train_classifier(self, num_of_epochs):

        train_file = open(self.log_dir+self.ltl_formula_string+"_train_acc_NS_exp"+str(self.exp_num), 'w')
        test_clss_file = open(self.log_dir+self.ltl_formula_string+"_test_clss_acc_NS_exp"+str(self.exp_num), 'w')
        test_aut_file = open(self.log_dir+self.ltl_formula_string+"_test_aut_acc_NS_exp"+str(self.exp_num), 'w')
        test_hard_file = open(self.log_dir+self.ltl_formula_string+"_test_hard_acc_NS_exp"+str(self.exp_num), 'w')
        image_classification_train_file = open(self.log_dir+self.ltl_formula_string+"_image_classification_train_acc_NS_exp"+str(self.exp_num), 'w')
        image_classification_test_file = open(self.log_dir+self.ltl_formula_string+"_image_classification_test_acc_NS_exp"+str(self.exp_num), 'w')
        self.classifier.to(device)
        self.deepAutoma.to(device)

        print("_____________training the classifier_____________")
        loss_final = final_states_loss
        if self.automa_implementation == 'lstm':
            params = [self.classifier.parameters(), self.deepAutoma.parameters()]
            params = itertools.chain(*params)
        else:
            params = self.classifier.parameters()
        optimizer = torch.optim.Adam(params=params, lr=0.001)

        batch_size = 64
        tot_size = len(self.train_img_seq)

        for epoch in range(num_of_epochs):
            print("epoch: ", epoch)
            for b in range(math.floor(tot_size/batch_size)):
                start = batch_size*b
                end = min(batch_size*(b+1), tot_size)
                batch_image_dataset = self.train_img_seq[start:end]
                batch_acceptance = self.train_acceptance_img[start:end]
                optimizer.zero_grad()
                losses_f = torch.zeros(0 ).to(device)
                losses_c = torch.zeros(0 ).to(device)

                for i in range(len(batch_image_dataset)):
                    img_sequence =batch_image_dataset[i].to(device)
                    target = batch_acceptance[i]
                    sym_sequence = self.classifier(img_sequence)
                    if self.automa_implementation == 'lstm':
                        states_sequence = self.deepAutoma.predict(sym_sequence)
                        final_state = states_sequence[-1]
                    else:
                        final_state = self.deepAutoma(sym_sequence)
                    if target == 0: #sequenza NON accettata
                        loss_f = not_final_states_loss(self.final_states, final_state)
                    else:
                        loss_f = loss_final(self.final_states, final_state)
                    losses_f = torch.cat((losses_f, loss_f.unsqueeze(dim=0)), 0)

                loss = losses_f.mean()

                if self.automa_implementation == 'lstm':
                    loss += losses_c.mean()

                loss.backward()
                optimizer.step()



            train_accuracy, test_accuracy_clss, test_accuracy_aut, test_accuracy_hard = self.eval_automa_acceptance(automa_implementation='dfa')
            print("__________________________")
            print("SEQUENCE CLASSIFICATION (DFA): train accuracy : {}\ttest accuracy(clss) : {}\ttest accuracy(aut) : {}\ttest accuracy(hard) : {}".format(train_accuracy,
                                                                                                 test_accuracy_clss, test_accuracy_aut, test_accuracy_hard))

            train_accuracy, test_accuracy_clss, test_accuracy_aut, test_accuracy_hard = self.eval_automa_acceptance(automa_implementation='logic_circuit')
            print("SEQUENCE CLASSIFICATION (LOGIC CIRCUIT): train accuracy : {}\ttest accuracy(clss) : {}\ttest accuracy(aut) : {}\ttest accuracy(hard) : {}".format(train_accuracy,
                                                                                                 test_accuracy_clss, test_accuracy_aut, test_accuracy_hard))
            train_image_classification_accuracy, test_image_classification_accuracy = self.eval_image_classification()
            print("IMAGE CLASSIFICATION: train accuracy : {}\ttest accuracy : {}".format(train_image_classification_accuracy,test_image_classification_accuracy))

            train_file.write("{}\n".format(train_accuracy))
            test_clss_file.write("{}\n".format(test_accuracy_clss))
            test_aut_file.write("{}\n".format(test_accuracy_aut))
            test_hard_file.write("{}\n".format(test_accuracy_hard))
            image_classification_train_file.write("{}\n".format(train_image_classification_accuracy))
            image_classification_test_file.write("{}\n".format(test_image_classification_accuracy))


    def train_classifier_crossentropy(self, num_of_epochs):
        train_file = open(self.log_dir+self.ltl_formula_string+"_train_acc_DL_exp"+str(self.exp_num), 'w')
        test_clss_file = open(self.log_dir+self.ltl_formula_string+"_test_clss_acc_DL_exp"+str(self.exp_num), 'w')
        test_aut_file = open(self.log_dir+self.ltl_formula_string+"_test_aut_acc_DL_exp"+str(self.exp_num), 'w')
        test_hard_file = open(self.log_dir+self.ltl_formula_string+"_test_hard_acc_DL_exp"+str(self.exp_num), 'w')
        print("_____________training classifier+lstm_____________")
        loss_crit = torch.nn.CrossEntropyLoss()
        params = [self.classifier.parameters(), self.deepAutoma.parameters()]
        params = itertools.chain(*params)
        optimizer = torch.optim.Adam(params=params, lr=0.001)
        batch_size = 64
        tot_size = len(self.train_img_seq)
        self.classifier.to(device)
        self.deepAutoma.to(device)

        for epoch in range(num_of_epochs):
            print("epoch: ", epoch)
            img_i =0
            for b in range(math.floor(tot_size/batch_size)):
                start = batch_size*b
                end = min(batch_size*(b+1), tot_size)
                batch_image_dataset = self.train_img_seq[start:end]
                batch_acceptance = self.train_acceptance_img[start:end]
                optimizer.zero_grad()
                losses = torch.zeros(0 ).to(device)


                for i in range(len(batch_image_dataset)):
                    img_sequence =batch_image_dataset[i].to(device)
                    target = batch_acceptance[i]
                    target = torch.LongTensor([target]).to(device)
                    sym_sequence = self.classifier(img_sequence)
                    acceptance = self.deepAutoma.predict(sym_sequence)
                    # Compute the loss, gradients, and update the parameters by
                    #  calling optimizer.step()
                    loss = loss_crit(acceptance.unsqueeze(0), target)
                    losses = torch.cat((losses, loss.unsqueeze(dim=0)), 0)

                loss = losses.mean()
                loss.backward()
                optimizer.step()
                #print("batch {}\tloss {}".format(b, loss))
            train_accuracy, test_accuracy_clss, test_accuracy_aut, test_accuracy_hard = self.eval_automa_acceptance(automa_implementation='lstm')
            print("__________________________train accuracy : {}\ttest accuracy(clss) : {}\ttest accuracy(aut) : {}\ttest accuracy(hard) : {}".format(train_accuracy,
                                                                                                 test_accuracy_clss, test_accuracy_aut, test_accuracy_hard))


            train_file.write("{}\n".format(train_accuracy))
            test_clss_file.write("{}\n".format(test_accuracy_clss))
            test_aut_file.write("{}\n".format(test_accuracy_aut))
            test_hard_file.write("{}\n".format(test_accuracy_hard))



