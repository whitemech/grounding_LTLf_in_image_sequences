from LTL_grounding import LTL_grounding
import random
import absl.flags
import absl.app
from utils import set_seed

from flloat.parser.ltlf import LTLfParser
from create_dataset import create_complete_set_traces_one_true_literal, create_image_sequence_dataset, create_complete_set_traces, create_image_sequence_dataset_non_mut_ex
import torchvision

from declare_formulas import formulas, formulas_names
from plot import plot_results, plot_results_all_formulas
import os

#flags
absl.flags.DEFINE_integer("MAX_LENGTH_TRACES", 4, "maximum traces length used to create the dataset")
absl.flags.DEFINE_float("TRAIN_SIZE_TRACES", 0.4, "portion of traces used for training")
absl.flags.DEFINE_bool("TRAIN_ON_RESTRICTED_DATASET", False, "if True test images from MNIST are used to render symbols")
absl.flags.DEFINE_bool("MUTUALLY_EXCLUSIVE_SYMBOLS", True, "if True symbols are mutually exclusive in traces")
absl.flags.DEFINE_string("LOG_DIR", "Results/", "path to save the results")
absl.flags.DEFINE_string("PLOTS_DIR", "Plots/", "path to save the plots")


FLAGS = absl.flags.FLAGS

def test_neurosym(formula, dfa, mutex, symbolic_dataset, image_seq_dataset, num_exp, epochs=20, log_dir="Results/"):
    ltl_ground =LTL_grounding(formula,dfa, mutex, symbolic_dataset, image_seq_dataset, 2, 100, 4, train_with_accepted_only=False, num_exp=num_exp, log_dir=log_dir)

    ltl_ground.train_classifier(epochs)

def test_supervised_learn(formula, dfa, mutex, symbolic_dataset, image_seq_dataset, num_exp, epochs=20, log_dir="Results/"):
    ltl_ground = LTL_grounding(formula, dfa, mutex, symbolic_dataset,image_seq_dataset, 2, 100, 4, train_with_accepted_only=False, automa_implementation= 'lstm', lstm_output="acceptance", num_exp=num_exp, log_dir=log_dir)

    ltl_ground.train_classifier_crossentropy(epochs)

############################################# EXPERIMENTS ######################################################################
def main(argv):
    if not os.path.isdir(FLAGS.LOG_DIR):
        os.makedirs(FLAGS.LOG_DIR)
    if not os.path.isdir(FLAGS.PLOTS_DIR):
        os.makedirs(FLAGS.PLOTS_DIR)
    #take the images
    normalize = torchvision.transforms.Normalize(mean=(0.1307,),
                                                     std=(0.3081,))

    transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            normalize,
        ])
    if FLAGS.TRAIN_ON_RESTRICTED_DATASET:
            test_data = torchvision.datasets.MNIST('./data/', train=True, download=True, transform=transforms)
            train_data = torchvision.datasets.MNIST('./data/', train=False, download=True, transform=transforms)
    else:
            train_data = torchvision.datasets.MNIST('./data/', train=True, download=True, transform=transforms)
            test_data = torchvision.datasets.MNIST('./data/', train=False, download=True, transform=transforms)

    num_exp = 10

    for i in range(len(formulas)):
        formula = formulas[i]
        formula_name = formulas_names[i]

        # translate the formula
        parser = LTLfParser()
        ltl_formula_parsed = parser(formula)
        dfa = ltl_formula_parsed.to_automaton()
        alphabet = ["c" + str(i) for i in range(2)]
        ###############create the dataset
        #set the seed for the random split of the dataset
        random.seed(1)

        ##### SYMBOLIC traces dataset
        if FLAGS.MUTUALLY_EXCLUSIVE_SYMBOLS:
            train_traces_tf, test_traces_tf, train_traces, test_traces, train_acceptance_tr, test_acceptance_tr = create_complete_set_traces_one_true_literal(FLAGS.MAX_LENGTH_TRACES, alphabet, dfa, train_with_accepted_only=False,
            train_size=FLAGS.TRAIN_SIZE_TRACES)
        else:
            _, _, train_traces, test_traces, train_acceptance_tr, test_acceptance_tr = create_complete_set_traces(
                FLAGS.MAX_LENGTH_TRACES, alphabet, dfa, train_with_accepted_only=False,
                train_size=FLAGS.TRAIN_SIZE_TRACES)
        symbolic_dataset = (train_traces, test_traces, train_acceptance_tr, test_acceptance_tr)

        ##### IMAGE traces dataset
        if FLAGS.MUTUALLY_EXCLUSIVE_SYMBOLS:
            # training dataset
            print("Training dataset")
            train_img_seq, train_acceptance_img = create_image_sequence_dataset(train_data, 2, train_traces,
                                                                                          train_acceptance_tr, print_size=True)
            # test clss
            test_img_seq_clss, test_acceptance_img_clss = create_image_sequence_dataset(test_data, 2,
                                                                                                  train_traces,
                                                                                                  train_acceptance_tr)
            # test_aut
            test_img_seq_aut, test_acceptance_img_aut = create_image_sequence_dataset(train_data, 2,
                                                                                                test_traces,
                                                                                                test_acceptance_tr)
            # test_hard
            print("Test dataset")
            test_img_seq_hard, test_acceptance_img_hard = create_image_sequence_dataset(test_data, 2,
                                                                                                  test_traces,
                                                                                                  test_acceptance_tr, print_size=True)

        else:
            # training dataset
            print("Training dataset")
            train_img_seq, train_acceptance_img = create_image_sequence_dataset_non_mut_ex(train_data, 2, train_traces,
                                                                                          train_acceptance_tr,print_size=True)
            # test clss
            test_img_seq_clss, test_acceptance_img_clss = create_image_sequence_dataset_non_mut_ex(test_data, 2,
                                                                                                  train_traces,
                                                                                                  train_acceptance_tr)
            # test_aut
            test_img_seq_aut, test_acceptance_img_aut = create_image_sequence_dataset_non_mut_ex(train_data, 2,
                                                                                                test_traces,
                                                                                                test_acceptance_tr)
            # test_hard
            print("Test dataset")
            test_img_seq_hard, test_acceptance_img_hard = create_image_sequence_dataset_non_mut_ex(test_data, 2,
                                                                                                  test_traces,
                                                                                                  test_acceptance_tr,print_size=True)

        image_seq_dataset = (train_img_seq, train_acceptance_img, test_img_seq_clss, test_acceptance_img_clss, test_img_seq_aut, test_acceptance_img_aut, test_img_seq_hard, test_acceptance_img_hard)

        for i in range(num_exp):
            #set_seed
            set_seed(9+i)
            print("###################### NEW TEST ###########################")
            print("formula = {},\texperiment = {}".format(formula, i))

            test_neurosym(formula, dfa, FLAGS.MUTUALLY_EXCLUSIVE_SYMBOLS, symbolic_dataset, image_seq_dataset, i, log_dir=FLAGS.LOG_DIR)

            test_supervised_learn(formula, dfa, FLAGS.MUTUALLY_EXCLUSIVE_SYMBOLS, symbolic_dataset, image_seq_dataset, i, log_dir=FLAGS.LOG_DIR)

        plot_results(formula, formula_name, res_dir = FLAGS.LOG_DIR,num_exp=num_exp, plot_legend=True, plot_dir= FLAGS.PLOTS_DIR)

    plot_results_all_formulas(formulas,dir=FLAGS.LOG_DIR, num_exp=num_exp,plot_legend=True, plot_dir= FLAGS.PLOTS_DIR)

if __name__ == '__main__':
    absl.app.run(main)


