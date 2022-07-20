import matplotlib.pyplot as plt
import numpy as np
import seaborn
import pandas as pd
def plot_results(formula, formula_name, res_dir='Results/', num_exp=2, plot_legend=False, plot_dir="Plots/"):
    experiments_to_keep = 8
    fontsize = 20
    train_rr = []
    train_img_rr = []
    test_img_rr = []
    test_hard_rr = []
    x = []

    train_rr_DL = []
    test_hard_rr_DL = []

    for i in range(num_exp):

        #risultati DL
        with open(res_dir+formula + "_train_acc_DL_exp"+str(i), 'r') as train_file:
            train_res = train_file.read().splitlines()
        train_res = [float(r) for r in train_res]
        train_rr_DL.append(train_res)

        with open(res_dir+formula + "_test_hard_acc_DL_exp"+str(i), 'r') as test_hard_file:
            test_hard_res = test_hard_file.read().splitlines()
        test_hard_res = [float(r) for r in test_hard_res]
        test_hard_rr_DL.append(test_hard_res)

        #results NS
        with open(res_dir+formula + "_train_acc_NS_exp"+str(i), 'r') as train_file:
            train_res = train_file.read().splitlines()
        train_res = [float(r) for r in train_res]
        train_rr.append(train_res)

        with open(res_dir+formula + "_test_hard_acc_NS_exp"+str(i), 'r') as test_hard_file:
            test_hard_res = test_hard_file.read().splitlines()
        test_hard_res = [float(r) for r in test_hard_res]

        test_hard_rr.append(test_hard_res)

        #Risultati image classification NS
        with open(res_dir+formula + "_image_classification_train_acc_NS_exp"+str(i), 'r') as train_file:
            train_res = train_file.read().splitlines()
        train_res = [50.0 + abs( float(r) - 50.0) for r in train_res]
        train_img_rr.append(train_res)

        with open(res_dir+formula + "_image_classification_test_acc_NS_exp"+str(i), 'r') as test_hard_file:
            test_hard_res = test_hard_file.read().splitlines()
        test_hard_res = [50.0 + abs( float(r) - 50.0) for r in test_hard_res]

        test_img_rr.append(test_hard_res)

    ############# eliminate outlayers NS
    dict_exp_to_keep = {}
    for i, res in enumerate(train_rr):
        dict_exp_to_keep[i] =res[-1]

    ordered = dict(sorted(dict_exp_to_keep.items(), key=lambda item: item[1]))

    keys = list(ordered.keys())

    keys = keys[-experiments_to_keep:]

    train_rr_no_ol = []
    test_hard_rr_no_ol = []
    train_img_rr_no_ol = []
    test_img_rr_no_ol = []

    for k in keys:
        train_rr_no_ol = train_rr_no_ol + train_rr[k]
        test_hard_rr_no_ol = test_hard_rr_no_ol + test_hard_rr[k]
        train_img_rr_no_ol = train_img_rr_no_ol + train_img_rr[k]
        test_img_rr_no_ol = test_img_rr_no_ol + test_img_rr[k]
        x = x + list(range(len(train_rr[k])))

    ############# eliminate outlayers DL
    dict_exp_to_keep = {}
    for i, res in enumerate(train_rr_DL):
        dict_exp_to_keep[i] = res[-1]


    ordered = dict(sorted(dict_exp_to_keep.items(), key=lambda item: item[1]))

    keys = list(ordered.keys())

    keys = keys[-experiments_to_keep:]

    train_rr_DL_no_ol = []
    test_hard_rr_DL_no_ol = []


    for k in keys:
        train_rr_DL_no_ol = train_rr_DL_no_ol + train_rr_DL[k]
        test_hard_rr_DL_no_ol = test_hard_rr_DL_no_ol + test_hard_rr_DL[k]


    #################à plot sequence classification
    plt.rcParams["figure.figsize"] = [6, 4.5]
    plt.rcParams["figure.autolayout"] = True
    fig, ax = plt.subplots()
    #train
    plot_line(x.copy(), train_rr_no_ol, "Train NS", plot_legend)
    plot_line(x.copy(), train_rr_DL_no_ol, "Train DL", plot_legend)

    plot_line(x.copy(), test_hard_rr_no_ol, "Test NS", plot_legend)
    plot_line(x.copy(), test_hard_rr_DL_no_ol, "Test DL", plot_legend)

    if plot_legend:
        plt.legend( prop={"size":fontsize})
    plt.title(formula_name,  fontdict={'fontsize': fontsize+3})
    #plt.xlabel("Epochs")
    #plt.ylabel("Sequence classification accuracy")
    ax.tick_params(axis='x', labelsize=fontsize)
    ax.tick_params(axis='y', labelsize=fontsize)
    plt.savefig(plot_dir+formula_name+"NSvsDL.png")
    plt.clf()

    ################### plot image classification
    #train
    plt.rcParams["figure.figsize"] = [6, 4.5]
    plt.rcParams["figure.autolayout"] = True
    fig, ax = plt.subplots()
    plot_line(x.copy(), train_img_rr_no_ol, "Train", plot_legend)

    #test
    plot_line(x.copy(), test_img_rr_no_ol, "Test", plot_legend)

    if plot_legend:
        plt.legend( prop={"size":fontsize})
    plt.title(formula_name, fontdict={'fontsize': fontsize+3})
    #plt.xlabel("Epochs")
    #plt.ylabel("Image classification accuracy")
    ax.tick_params(axis='x', labelsize=fontsize)
    ax.tick_params(axis='y', labelsize=fontsize)
    plt.savefig(plot_dir+formula_name+"_image_classification.png")
    plt.clf()
    return


def plot_results_all_formulas(formulas, dir='Results/', num_exp=2, plot_legend=False, plot_dir="Plots/"):
    experiments_to_keep = 8
    fontsize = 20
    formula_name = "Mean over 20 Declare formulas"
    train_rr_no_ol_cum = []
    test_hard_rr_no_ol_cum = []
    train_img_rr_no_ol_cum = []
    test_img_rr_no_ol_cum = []
    train_rr_DL_no_ol_cum = []
    test_hard_rr_DL_no_ol_cum = []
    x_cum = []

    for formula in formulas:
        train_rr = []
        train_img_rr = []
        test_img_rr = []

        test_hard_rr = []
        x = []

        train_rr_DL = []

        test_hard_rr_DL = []

        for i in range(num_exp):

            #risultati DL
            with open(dir+formula + "_train_acc_DL_exp"+str(i), 'r') as train_file:
                train_res = train_file.read().splitlines()
            train_res = [float(r) for r in train_res]
            train_rr_DL.append(train_res)

            with open(dir+formula + "_test_hard_acc_DL_exp"+str(i), 'r') as test_hard_file:
                test_hard_res = test_hard_file.read().splitlines()
            test_hard_res = [float(r) for r in test_hard_res]
            test_hard_rr_DL.append(test_hard_res)
            ###############################################################################
            #risultati NS
            with open(dir+formula + "_train_acc_NS_exp"+str(i), 'r') as train_file:
                train_res = train_file.read().splitlines()
            train_res = [float(r) for r in train_res]
            train_rr.append(train_res)

            with open(dir+formula + "_test_hard_acc_NS_exp"+str(i), 'r') as test_hard_file:
                test_hard_res = test_hard_file.read().splitlines()
            test_hard_res = [float(r) for r in test_hard_res]

            test_hard_rr.append(test_hard_res)

            #Risultati image classification NS
            with open(dir+formula + "_image_classification_train_acc_NS_exp"+str(i), 'r') as train_file:
                train_res = train_file.read().splitlines()
            train_res = [50.0 + abs( float(r) - 50.0) for r in train_res]
            train_img_rr.append(train_res)

            with open(dir+formula + "_image_classification_test_acc_NS_exp"+str(i), 'r') as test_hard_file:
                test_hard_res = test_hard_file.read().splitlines()
            test_hard_res = [50.0 + abs( float(r) - 50.0) for r in test_hard_res]

            test_img_rr.append(test_hard_res)

        ############# eliminate outlayers NS
        dict_exp_to_keep = {}
        for i, res in enumerate(train_rr):
            dict_exp_to_keep[i] =res[-1]

        ordered = dict(sorted(dict_exp_to_keep.items(), key=lambda item: item[1]))

        keys = list(ordered.keys())
        keys = keys[-experiments_to_keep:]

        train_rr_no_ol = []
        test_hard_rr_no_ol = []
        train_img_rr_no_ol = []
        test_img_rr_no_ol = []

        for k in keys:
            train_rr_no_ol = train_rr_no_ol + train_rr[k]
            test_hard_rr_no_ol = test_hard_rr_no_ol + test_hard_rr[k]
            train_img_rr_no_ol = train_img_rr_no_ol + train_img_rr[k]
            test_img_rr_no_ol = test_img_rr_no_ol + test_img_rr[k]
            x = x + list(range(len(train_rr[k])))

        ############# eliminate outlayers DL
        dict_exp_to_keep = {}
        for i, res in enumerate(train_rr_DL):
            dict_exp_to_keep[i] = res[-1]

        ordered = dict(sorted(dict_exp_to_keep.items(), key=lambda item: item[1]))

        keys = list(ordered.keys())
        keys = keys[-experiments_to_keep:]

        train_rr_DL_no_ol = []
        test_hard_rr_DL_no_ol = []


        for k in keys:
            train_rr_DL_no_ol = train_rr_DL_no_ol + train_rr_DL[k]
            test_hard_rr_DL_no_ol = test_hard_rr_DL_no_ol + test_hard_rr_DL[k]

        train_rr_no_ol_cum = train_rr_no_ol_cum + train_rr_no_ol
        test_hard_rr_no_ol_cum = test_hard_rr_no_ol_cum + test_hard_rr_no_ol
        train_img_rr_no_ol_cum = train_img_rr_no_ol_cum + train_img_rr_no_ol
        test_img_rr_no_ol_cum = test_img_rr_no_ol_cum + test_img_rr_no_ol
        train_rr_DL_no_ol_cum = train_rr_DL_no_ol_cum + train_rr_DL_no_ol
        test_hard_rr_DL_no_ol_cum = test_hard_rr_DL_no_ol_cum + test_hard_rr_DL_no_ol
        x_cum = x_cum + x

    #################à plot sequence classification
    plt.rcParams["figure.figsize"] = [8, 4.5]
    plt.rcParams["figure.autolayout"] = True
    fig, ax = plt.subplots()
    #train
    plot_line(x_cum.copy(), train_rr_no_ol_cum, "Train NS", plot_legend)
    plot_line(x_cum.copy(), train_rr_DL_no_ol_cum, "Train DL", plot_legend)

    plot_line(x_cum.copy(), test_hard_rr_no_ol_cum, "Test NS", plot_legend)
    plot_line(x_cum.copy(), test_hard_rr_DL_no_ol_cum, "Test DL", plot_legend)

    if plot_legend:
        plt.legend( prop={"size":fontsize-2}, bbox_to_anchor =(1, 1))
    plt.title("Sequence classification accuracy",  fontdict={'fontsize': fontsize+3})
    plt.xlabel("Epochs", fontsize=fontsize-2)
    #plt.ylabel("Sequence classification accuracy", fontsize=fontsize-2)
    ax.tick_params(axis='x', labelsize=fontsize)
    ax.tick_params(axis='y', labelsize=fontsize)
    plt.savefig(plot_dir+formula_name+"NSvsDL.png")
    plt.clf()

    ################### plot image classification
    #train
    plt.rcParams["figure.figsize"] = [8, 4.5]
    plt.rcParams["figure.autolayout"] = True
    fig, ax = plt.subplots()
    plot_line(x_cum.copy(), train_img_rr_no_ol_cum, "Train", plot_legend)

    #test
    plot_line(x_cum.copy(), test_img_rr_no_ol_cum, "Test", plot_legend)

    if plot_legend:
        plt.legend( prop={"size":fontsize-2}, bbox_to_anchor =(1, 1))
    plt.title("Image classification accuracy", fontdict={'fontsize': fontsize+3})
    plt.xlabel("Epochs", fontsize=fontsize-2)
    ax.tick_params(axis='x', labelsize=fontsize)
    ax.tick_params(axis='y', labelsize=fontsize)
    plt.savefig(plot_dir+formula_name+"_image_classification.png")
    plt.clf()
    return


def plot_line(x, y, label, plot_legend=False):
    data = [x, y]
    data = np.array(data)
    data = data.T
    data = pd.DataFrame(data, columns=['x', 'y'])
    if plot_legend:
        line = seaborn.lineplot( x = data['x'],y=data['y'], label=label)
    else:
        line = seaborn.lineplot( x = data['x'],y=data['y'])
    return