import json
import os
from time import time

import fire
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import LearningCurveDisplay
from sklearn.model_selection import ShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def experiments_default(eval=False):
    default = (None, 0., "") if eval else []
    return {
        DecisionTreeClassifier: default,
        MLPClassifier: default,
        AdaBoostClassifier: default,
        SVC: default,
        KNeighborsClassifier: default,
    }


def classifier_eval(model, X_train, y_train, X_val, y_val):
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    return y_train_pred, y_val_pred, {
        "train": {
            "accuracy": accuracy_score(y_train, y_train_pred),
            "precision": precision_score(y_train, y_train_pred, average="weighted"),
            "recall": recall_score(y_train, y_train_pred, average="weighted"),
            "f1": f1_score(y_train, y_train_pred, average="weighted"),
        },
        "val": {
            "accuracy": accuracy_score(y_val, y_val_pred),
            "precision": precision_score(y_val, y_val_pred, average="weighted"),
            "recall": recall_score(y_val, y_val_pred, average="weighted"),
            "f1": f1_score(y_val, y_val_pred, average="weighted"),
        },
    }


def plot_cm(y_train, y_train_pred, classes, filename, model_type, experiment_name):
    plt.figure(figsize=(10, 6))
    cf = confusion_matrix(y_train, y_train_pred)
    sns.heatmap(
        cf,
        annot=True,
        yticklabels=classes,
        xticklabels=classes,
        cmap="Greens",
        fmt="g"
    )
    plt.tight_layout()

    file_basename = f"plots/{model_type.lower()}/{experiment_name}"
    if not os.path.exists(file_basename):
        os.makedirs(file_basename)
    plt.savefig(f"{file_basename}/{filename}.png")


def plot_learning_curve(X, y, model_type, model, mode, experiment_name, seed):
    common_params = {
        "X": X,
        "y": y,
        "train_sizes": np.linspace(0.1, 1.0, 5),
        "cv": ShuffleSplit(n_splits=50, test_size=0.2, random_state=seed),
        "score_type": "both",
        "n_jobs": 16,
        "line_kw": {"marker": "o"},
        "std_display_style": "fill_between",
        "score_name": "Accuracy",
    }
    LearningCurveDisplay.from_estimator(model, **common_params)
    file_basename = f"plots/{model_type.lower()}/{experiment_name}"
    if not os.path.exists(file_basename):
        os.makedirs(file_basename)
    plt.savefig(f"{file_basename}/{mode}-learning-curve.png")


def plot_loss(model, mode, experiment_name):
    plt.figure(figsize=(10, 6))
    plt.plot(model.loss_curve_)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    file_basename = f"plots/nn/{experiment_name}"
    if not os.path.exists(file_basename):
        os.makedirs(file_basename)
    plt.savefig(f"{file_basename}/{mode}-loss.png")


def plot_exp_loss(experiments, mode, exp=1):
    fig, ax = plt.subplots()
    for experiment in experiments:
        if exp == 1:
            metric_name = "Number of hidden layers"
            metric = len(experiment["kwargs"]["hidden_layer_sizes"])
        elif exp == 2:
            metric_name = "Width of hidden layers"
            metric = experiment["kwargs"]["hidden_layer_sizes"][0]
        elif exp == 3:
            metric_name = "Activation"
            metric = experiment["kwargs"]["activation"]

        plt.plot(experiment["model"].loss_curve_, label=metric)

    ax.set_title(f"Neural Network Tuning - Losses for {metric_name}")
    fig.tight_layout()
    plt.legend()
    file_basename = f"plots/nn"
    if not os.path.exists(file_basename):
        os.makedirs(file_basename)
    plt.savefig(f"{file_basename}/{mode}-loss-all-exp-{exp}.png")


def plot_feature_importance(model, model_type, mode, experiment_name):
    feature_importances = pd.Series(model.feature_importances_)
    fig, ax = plt.subplots()
    feature_importances.plot.bar(ax=ax)
    ax.set_title("Feature importances")
    fig.tight_layout()
    file_basename = f"plots/{model_type}/{experiment_name}"
    if not os.path.exists(file_basename):
        os.makedirs(file_basename)
    plt.savefig(f"{file_basename}/{mode}-feature-importance.png")


def plot_mean_feature_importance(experiments, mode):
    fig, ax = plt.subplots()
    feature_importances = [e["model"].feature_importances_ for e in experiments]
    feature_importances = np.array(feature_importances)
    feature_importances_mean = pd.Series(np.mean(feature_importances, axis=0))
    feature_importances_std = np.std(feature_importances, axis=0)
    feature_importances_mean.plot.bar(yerr=feature_importances_std, ax=ax)
    ax.set_title("Mean Feature Importances - All Experiments")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
    file_basename = f"plots/dt"
    if not os.path.exists(file_basename):
        os.makedirs(file_basename)
    plt.savefig(f"{file_basename}/{mode}-feature-importance-all.png")


def plot_dt_pruning(experiments, mode, X_train, y_train, X_val, y_val):
    file_basename = f"plots/dt"
    if not os.path.exists(file_basename):
        os.makedirs(file_basename)
    assert len(experiments) == 1
    model = experiments[0]["model"]
    path = model.cost_complexity_pruning_path(X_train, y_train)
    plt.figure(figsize=(10, 6))
    plt.plot(path.ccp_alphas, path.impurities)
    plt.xlabel("effective alpha")
    plt.ylabel("total impurity of leaves")
    plt.savefig(f"{file_basename}/{mode}-impurity-pruned.png")

    pruned = []
    for ccp_alpha in path.ccp_alphas:
        pruned_eg = DecisionTreeClassifier(ccp_alpha=ccp_alpha, **experiments[0]["kwargs"])
        pruned_eg.fit(X_train, y_train)
        pruned.append(pruned_eg)

    pruned_ = pruned[:-1]
    ccp_alphas_ = path.ccp_alphas[:-1]

    node_counts = [m.tree_.node_count for m in pruned_]
    depth = [m.tree_.max_depth for m in pruned_]
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(ccp_alphas_, node_counts, marker="o", drawstyle="steps-post")
    ax[0].set_xlabel("alpha")
    ax[0].set_ylabel("number of nodes")
    ax[0].set_title("Number of nodes vs alpha")
    ax[1].plot(ccp_alphas_, depth, marker="o", drawstyle="steps-post")
    ax[1].set_xlabel("alpha")
    ax[1].set_ylabel("depth of tree")
    ax[1].set_title("Depth vs alpha")
    fig.tight_layout()

    tree_depths = [m.tree_.max_depth for m in pruned]
    plt.figure(figsize=(10,  6))
    plt.plot(path.ccp_alphas[:-1], tree_depths[:-1])
    plt.xlabel("effective alpha")
    plt.ylabel("total depth")
    plt.savefig(f"{file_basename}/{mode}-depths.png")

    acc_scores = [accuracy_score(y_val, m.predict(X_val)) for m in pruned]

    tree_depths = [m.tree_.max_depth for m in pruned]
    plt.figure(figsize=(10,  6))
    plt.grid()
    plt.plot(path.ccp_alphas[:-1], acc_scores[:-1])
    plt.xlabel("effective alpha")
    plt.ylabel("Accuracy scores")
    plt.savefig(f"{file_basename}/{mode}-alphas.png")


def plot_dt_compare(experiments, mode, exp=1):
    X = []
    y_a = []
    y_a_v = []
    for experiment in experiments:
        if exp == 1:
            metric_name = "Number of hidden layers"
            metric = len(experiment["kwargs"]["hidden_layer_sizes"])
        elif exp == 2:
            metric_name = "Width of hidden layers"
            metric = experiment["kwargs"]["hidden_layer_sizes"][0]
        elif exp == 3:
            metric_name = "Activation"
            metric = experiment["kwargs"]["activation"]

        X.append(metric)
        y_a_v.append(experiment["metrics"]["val"]["accuracy"])
        y_a.append(experiment["metrics"]["train"]["accuracy"])

    plt.figure(figsize=(10, 6))
    plt.plot(X, y_a_v, label="Accuracy (validation)")
    plt.plot(X, y_a, label="Accuracy (train)")
    plt.xlabel(metric_name)
    plt.title(f"Neural Network Tuning - {metric_name}")
    plt.legend()
    file_basename = f"plots/nn"
    if not os.path.exists(file_basename):
        os.makedirs(file_basename)
    plt.savefig(f"{file_basename}/{mode}-nn-exp-{exp}.png") 


def plot_nn_compare(experiments, mode, exp=1):
    X = []
    y_a = []
    y_a_v = []
    for experiment in experiments:
        if exp == 1:
            plot_type = plt.plot
            metric_name = "Number of hidden layers"
            metric = len(experiment["kwargs"]["hidden_layer_sizes"])
        elif exp == 2:
            plot_type = plt.plot
            metric_name = "Width of hidden layers"
            metric = experiment["kwargs"]["hidden_layer_sizes"][0]
        elif exp == 3:
            plot_type = plt.bar
            metric_name = "Activation"
            metric = experiment["kwargs"]["activation"]

        X.append(metric)
        y_a_v.append(experiment["metrics"]["val"]["accuracy"])
        y_a.append(experiment["metrics"]["train"]["accuracy"])

    plt.figure(figsize=(10, 6))
    if exp == 3:
        bar_width = 0.33
        plot_type(np.arange(len(X)), y_a, bar_width, label="Accuracy (train)")
        plot_type(np.arange(len(X)) + bar_width, y_a_v, bar_width, label="Accuracy (validation)")
        plt.xticks(np.arange(len(X)) + bar_width / 2, X)
    else:
        plot_type(X, y_a_v, label="Accuracy (validation)")
        plot_type(X, y_a, label="Accuracy (train)")
        plt.xlabel(metric_name)
    plt.title(f"Neural Network Tuning - {metric_name}")
    plt.legend()
    file_basename = f"plots/nn"
    if not os.path.exists(file_basename):
        os.makedirs(file_basename)
    plt.savefig(f"{file_basename}/{mode}-exp-{exp}.png")


def plot_dt_compare(experiments, mode, exp=1):
    X = []
    y_a = []
    y_a_v = []
    for experiment in experiments:
        if exp == 1:
            plot_type = plt.bar
            metric_name = "Split Criteria"
            metric = experiment["kwargs"]["criterion"]
        if exp == 2:
            plot_type = plt.bar
            metric_name = "Max. Features"
            metric = experiment["kwargs"]["max_features"]

        X.append(metric)
        y_a_v.append(experiment["metrics"]["val"]["accuracy"])
        y_a.append(experiment["metrics"]["train"]["accuracy"])

    plt.figure(figsize=(10, 6))
    bar_width = 0.33
    plot_type(np.arange(len(X)), y_a, bar_width, label="Accuracy (train)")
    plot_type(np.arange(len(X)) + bar_width, y_a_v, bar_width, label="Accuracy (validation)")
    plt.xticks(np.arange(len(X)) + bar_width / 2, [x if x else "n_features" for x in X])
    plt.xlabel(metric_name)
    plt.title(f"Decision Tree Tuning - {metric_name}")
    plt.legend()
    file_basename = f"plots/dt"
    if not os.path.exists(file_basename):
        os.makedirs(file_basename)
    plt.savefig(f"{file_basename}/{mode}-exp-{exp}.png")


def plot_knn_compare(experiments, mode, exp=1):
    X = []
    y_a = []
    y_a_v = []
    for experiment in experiments:
        if exp == 1:
            plot_type = plt.bar
            metric_name = "K Neighbors"
            metric = experiment["kwargs"]["n_neighbors"]
        if exp == 2:
            plot_type = plt.plot
            metric_name = "P Distance"
            metric = experiment["kwargs"]["p"]
        if exp == 3:
            plot_type = plt.bar
            metric_name = "Weights"
            metric = experiment["kwargs"]["weights"]

        X.append(metric)
        y_a_v.append(experiment["metrics"]["val"]["accuracy"])
        y_a.append(experiment["metrics"]["train"]["accuracy"])

    plt.figure(figsize=(10, 6))
    if exp in [1, 3]:
        bar_width = 0.33
        plot_type(np.arange(len(X)), y_a, bar_width, label="Accuracy (train)")
        plot_type(np.arange(len(X)) + bar_width, y_a_v, bar_width, label="Accuracy (validation)")
        plt.xticks(np.arange(len(X)) + bar_width / 2, [x if x else "n_features" for x in X])
    if exp == 2:
        plot_type(X, y_a_v, label="Accuracy (validation)")
        plot_type(X, y_a, label="Accuracy (train)")
        plt.xlabel(metric_name)

    plt.xlabel(metric_name)
    plt.title(f"KNN Tuning - {metric_name}")
    plt.legend()
    file_basename = f"plots/knn"
    if not os.path.exists(file_basename):
        os.makedirs(file_basename)
    plt.savefig(f"{file_basename}/{mode}-exp-{exp}.png")


def plot_svm_compare(experiments, mode, exp=1):
    X = []
    y_a = []
    y_a_v = []
    for experiment in experiments:
        if exp == 1:
            plot_type = plt.bar
            metric_name = "Kernel Type"
            metric = experiment["kwargs"]["kernel"]
        if exp == 2:
            plot_type = plt.plot
            metric_name = "C"
            metric = experiment["kwargs"]["C"]

        X.append(metric)
        y_a_v.append(experiment["metrics"]["val"]["accuracy"])
        y_a.append(experiment["metrics"]["train"]["accuracy"])

    plt.figure(figsize=(10, 6))
    if exp == 1:
        bar_width = 0.33
        plot_type(np.arange(len(X)), y_a, bar_width, label="Accuracy (train)")
        plot_type(np.arange(len(X)) + bar_width, y_a_v, bar_width, label="Accuracy (validation)")
        plt.xticks(np.arange(len(X)) + bar_width / 2, [x if x else "n_features" for x in X])
    if exp == 2:
        plot_type(X, y_a_v, label="Accuracy (validation)")
        plot_type(X, y_a, label="Accuracy (train)")
        plt.xlabel(metric_name)

    plt.xlabel(metric_name)
    plt.title(f"SVM Tuning - {metric_name}")
    plt.legend()
    file_basename = f"plots/svm"
    if not os.path.exists(file_basename):
        os.makedirs(file_basename)
    plt.savefig(f"{file_basename}/{mode}-exp-{exp}.png")


def plot_boost_compare(experiments, mode, exp=1):
    X = []
    y_a = []
    y_a_v = []
    for experiment in experiments:
        if exp == 1:
            plot_type = plt.plot
            metric_name = "Number of learners"
            metric = experiment["kwargs"]["n_estimators"]
        if exp == 2:
            plot_type = plt.plot
            metric_name = "Learning rate"
            metric = experiment["kwargs"]["learning_rate"]

        X.append(metric)
        y_a_v.append(experiment["metrics"]["val"]["accuracy"])
        y_a.append(experiment["metrics"]["train"]["accuracy"])

    plt.figure(figsize=(10, 6))
    plot_type(X, y_a_v, label="Accuracy (validation)")
    plot_type(X, y_a, label="Accuracy (train)")
    plt.xlabel(metric_name)
    plt.title(f"Boosting Tuning - {metric_name}")
    plt.legend()
    file_basename = f"plots/boost"
    if not os.path.exists(file_basename):
        os.makedirs(file_basename)
    plt.savefig(f"{file_basename}/{mode}-exp-{exp}.png")


def create_datasets(seed, classes, n=2):
    n_samples = [732, 16144]
    n_features = [12, 18]
    n_redundant = [1, 3]
    datasets = []
    for (
        samples,
        features,
        redundant
    ) in zip(
        n_samples,
        n_features,
        n_redundant,
    ):
        x, y = make_classification(
            n_samples=samples,
            n_classes=classes,
            n_features=features,
            n_redundant=redundant,
            n_informative=features - redundant,
            n_clusters_per_class=1,
            random_state=seed,
        )
        datasets.append((x, y))

    return datasets


def method_short_name(method):
    return {
        DecisionTreeClassifier: "dt",
        MLPClassifier: "nn",
        AdaBoostClassifier: "boost",
        SVC: "svm",
        KNeighborsClassifier: "knn",
    }[method]


def experiment_namer(kwargs):
    return "-".join(f"{k}-{v}" for k, v in kwargs.items() if k != "random_state")


def instantiate(method, experiment, seed):
    if method_short_name(method) != "knn":
        experiment["kwargs"]["random_state"] = seed

    return method(**experiment["kwargs"])


def dichotomize(dataset_or_sets):
    if not isinstance(dataset_or_sets, (list, tuple, set)):
        dataset_or_sets = [dataset_or_sets]

    for dataset in dataset_or_sets:
        temp = dataset["placement"]
        temp = [1 if x == 1 else 0 for x in temp]
        dataset["placement"] = temp
        yield dataset


def main(
    run_dt=False,
    run_nn=False,
    run_boost=False,
    run_svm=False,
    run_knn=False,
    dt_experiment_1=False,
    dt_experiment_2=False,
    dt_experiment_3=False,
    dt_experiment_4=False,
    nn_experiment_1=False,
    nn_experiment_2=False,
    nn_experiment_3=False,
    nn_experiment_4=False,
    boost_experiment_1=False,
    boost_experiment_2=False,
    boost_experiment_3=False,
    svm_experiment_1=False,
    svm_experiment_2=False,
    svm_experiment_3=False,
    knn_experiment_1=False,
    knn_experiment_2=False,
    knn_experiment_3=False,
    knn_experiment_4=False,
    mode="real",
    seed=1560152,
):
    classes = 8
    datasets = create_datasets(classes=classes, seed=seed)

    dataset1, dataset2 = datasets
    X, y = dataset1
    X_s, y_s = dataset2

    dt_experiments = []
    if dt_experiment_1:
        for criterion in ["gini", "entropy", "log_loss"]:
            dt_experiments.append({
                "kwargs": {
                    "criterion": criterion,
                }
            })

    elif dt_experiment_2:
        for max_features in [None, "auto", "sqrt", "log2"]:
            dt_experiments.append({
                "kwargs": {
                    "criterion": "log_loss",
                    "max_features": max_features,
                }
            })

    elif dt_experiment_3:
        dt_experiments = [{
            "kwargs": {
                "criterion": "log_loss",
            }
        }]

    elif dt_experiment_4:
        if mode == "synth":
            ccp_alpha = 0.002
            max_depth = 17
        elif mode == "real":
            ccp_alpha = 0.025
            max_depth = 9

        dt_experiments = [{
            "kwargs": {
                "criterion": "log_loss",
                "ccp_alpha": ccp_alpha,
                "max_depth": max_depth,
            }
        }]

    mlp_experiments = []
    if nn_experiment_1:
        for n in range(1, 6):
            mlp_experiments.append({
                "kwargs": {
                    "hidden_layer_sizes": [100 for _ in range(n)],
                }
            })

    if nn_experiment_2:
        if mode == "synth":
            hidden_layer_sizes = lambda n: (n,)
        elif mode == "real":
            hidden_layer_sizes = lambda n: (n, n, n)

        for n in range(25, 650, 25):
            mlp_experiments.append({
                "kwargs": {
                    "hidden_layer_sizes": hidden_layer_sizes(n),
                }
            })

    if nn_experiment_3:
        if mode == "synth":
            hidden_layer_sizes = (600,)
        elif mode == "real":
            hidden_layer_sizes = (550, 550, 550)

        for activation in ["relu", "tanh", "identity", "logistic"]:
            mlp_experiments.append({
                "kwargs": {
                    "hidden_layer_sizes": hidden_layer_sizes,
                    "activation": activation,
                }
            })

    if nn_experiment_4:
        if mode == "synth":
            hidden_layer_sizes = (600,)
        elif mode == "real":
            hidden_layer_sizes = (550, 550, 550)

        mlp_experiments.append({
            "kwargs": {
                "hidden_layer_sizes": hidden_layer_sizes,
                "activation": "relu",
            }
        })

    boost_experiments = []
    if boost_experiment_1:
        if mode == "real":
            nrange = range(1, 33)
        elif mode == "synth":
            nrange = range(2, 252, 2)

        for n in nrange:
            boost_experiments.append({
                "kwargs": {
                    "n_estimators": n,
                }
            })

    if boost_experiment_2:
        if mode == "real":
            n = 16
        elif mode == "synth":
            n = 54

        for lr in np.arange(0.05, 1., 0.05):
            boost_experiments.append({
                "kwargs": {
                    "n_estimators": n,
                    "learning_rate": lr,
                }
            })

    if boost_experiment_3:
        if mode == "real":
            n = 16
            lr = 0.3
        elif mode == "synth":
            n = 54
            lr = 0.2

        boost_experiments.append({
            "kwargs": {
                "n_estimators": n,
                "learning_rate": lr,
            }
        })

    svm_experiments = []
    if svm_experiment_1:
        for kernel in ["linear", "poly", "rbf", "sigmoid"]:
            svm_experiments.append({
                "kwargs": {
                    "kernel": kernel,
                }
            })

    if svm_experiment_2:
        if mode == "real":
            kernel = "linear"
        elif mode == "synth":
            kernel = "rbf"

        for c in np.arange(1., 12., 1.):
            svm_experiments.append({
                "kwargs": {
                    "kernel": kernel,
                    "C": c,
                }
            })

    if svm_experiment_3:
        if mode == "real":
            kernel = "linear"
            C = 0.7
        elif mode == "synth":
            kernel = "rbf"
            C = 3.8

        svm_experiments.append({
            "kwargs": {
                "kernel": kernel,
                "C": C,
            }
        })

    knn_experiments = []
    if knn_experiment_1:
        for n in range(1, 33):
            knn_experiments.append({
                "kwargs": {
                    "n_neighbors": n,
                }
            })

    if knn_experiment_2:
        if mode == "real":
            n = 4
        elif mode == "synth":
            n = 10

        for p in np.arange(1., 11., 0.1):
            knn_experiments.append({
                "kwargs": {
                    "n_neighbors": n,
                    "p": p,
                }
            })

    if knn_experiment_3:
        if mode == "real":
            n = 4
            p = 1.7
        elif mode == "synth":
            n = 10
            p = 1.5

        for weights in ["uniform", "distance"]:
            knn_experiments.append({
                "kwargs": {
                    "n_neighbors": n,
                    "p": p,
                    "weights": weights,
                }
            })

    if knn_experiment_4:
        if mode == "real":
            n = 4
            p = 1.7
            weights = "distance"
        elif mode == "synth":
            n = 10
            p = 1.5
            weights = "distance"

        knn_experiments.append({
            "kwargs": {
                "n_neighbors": n,
                "p": p,
                "weights": weights,
            }
        })

    all_experiments = {}
    if run_dt:
        all_experiments[DecisionTreeClassifier] = dt_experiments
    if run_nn:
        all_experiments[MLPClassifier] = mlp_experiments
    if run_boost:
        all_experiments[AdaBoostClassifier] = boost_experiments
    if run_svm:
        all_experiments[SVC] = svm_experiments
    if run_knn:
        all_experiments[KNeighborsClassifier] = knn_experiments

    split_kwargs = dict(test_size=0.8, random_state=seed)
    val_split_kwargs = dict(test_size=0.5, random_state=seed)

    best_real = experiments_default(eval=True)

    if mode == "synth":
        X_, y_ = X_s, y_s
    else:
        X_, y_ = X, y

    X_train, X_val, y_train, y_val = train_test_split(X_, y_, **split_kwargs)
    X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, **val_split_kwargs)

    for method, experiments in tqdm(all_experiments.items()):
        for experiment in tqdm(experiments):
            model = instantiate(method, experiment, seed)
            method_name = method_short_name(method)

            start_time = time()
            model.fit(X_train, y_train)
            end_time = time()
            execution_time = end_time - start_time

            experiment_name = experiment_namer(experiment["kwargs"])
            y_train_pred, y_val_pred, metrics = classifier_eval(
                model,
                X_train,
                y_train,
                X_val,
                y_val,
            )
            metrics["execution_time"] = execution_time
            experiment["metrics"] = metrics
            experiment["model"] = model

            if dt_experiment_4 or nn_experiment_4 or boost_experiment_3 or svm_experiment_3 or knn_experiment_4:
                plot_cm(y_train, y_train_pred, classes, f"{mode}-cm-train", method_name, experiment_name)
                plot_cm(y_val, y_val_pred, classes, f"{mode}-cm-val", method_name, experiment_name)
                plot_learning_curve(X_, y_, method_name, model, mode, experiment_name, seed)

            if method_name == "dt":
                plot_feature_importance(model, method_name, mode, experiment_name)

            print(json.dumps(metrics, indent=4, default=str))

            if best_real[method][1] <= metrics["val"]["accuracy"]:
                best_real[method] = (model, metrics["val"]["accuracy"], experiment_name)

        if run_dt:
            if dt_experiment_1:
                exp = 1
            elif dt_experiment_2:
                exp = 2
            elif dt_experiment_3:
                exp = 3
                plot_dt_pruning(experiments, mode, X_train, y_train, X_val, y_val)

            if dt_experiment_1 or dt_experiment_2:
                plot_dt_compare(experiments, mode, exp=exp)
                plot_mean_feature_importance(experiments, mode)

        if run_nn:
            if nn_experiment_1:
                exp = 1
            elif nn_experiment_2:
                exp = 2
            elif nn_experiment_3:
                exp = 3

            if not nn_experiment_4:
                plot_exp_loss(experiments, mode, exp=exp)
                plot_nn_compare(experiments, mode, exp=exp)

        if run_boost:
            if boost_experiment_1:
                exp = 1
            elif boost_experiment_2:
                exp = 2

            if not boost_experiment_3:
                plot_boost_compare(experiments, mode, exp=exp)

        if run_svm:
            if svm_experiment_1:
                exp = 1
            elif svm_experiment_2:
                exp = 2
                # plot_svm_margins()

            if not svm_experiment_3:
                plot_svm_compare(experiments, mode, exp=exp)

        if run_knn:
            if knn_experiment_1:
                exp = 1
            elif knn_experiment_2:
                exp = 2
            elif knn_experiment_3:
                exp = 3

            if not knn_experiment_4:
                plot_knn_compare(experiments, mode, exp=exp)

    print(json.dumps(best_real.values(), indent=4, default=str))

    best_metrics = {}
    for method, (model, _, experiment_name) in best_real.items():
        if model is None:
            continue

        method_name = method_short_name(method)
        y_test_pred = model.predict(X_test)
        plot_cm(y_test, y_test_pred, classes, f"{mode}-cm-test", method_name, experiment_name)

        best_metrics[method_name] = {
            "accuracy": accuracy_score(y_test, y_test_pred),
            "precision": precision_score(y_test, y_test_pred, average="weighted"),
            "recall": recall_score(y_test, y_test_pred, average="weighted"),
            "f1": f1_score(y_test, y_test_pred, average="weighted"),
        }

    print(json.dumps(best_metrics, indent=4, default=str))


if __name__ == "__main__":
    fire.Fire(main)
