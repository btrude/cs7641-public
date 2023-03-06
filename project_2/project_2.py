import multiprocessing
import random
import six
import sys
import time
from functools import partial

sys.modules["sklearn.externals.six"] = six

import fire
import matplotlib.pyplot as plt
import mlrose_hiive
import networkx as nx
import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm


random.seed(24)
np.random.seed(24)
MAX_ATTEMPTS = 100
MAX_ITERS = 100


def metrics(name, score, context):
    with open("plots/metrics.csv", 'a+') as metric_file:
        metric_file.write(f"{name}|{round(score, 5)}|{context}\n")


def time_fn(opt_fn):
    start_time = time.time()
    _, _, fitness_curve = opt_fn()
    end_time = time.time()
    return fitness_curve, end_time - start_time


def queens():
    fitness_fn = mlrose_hiive.Queens()
    problem = mlrose_hiive.DiscreteOpt(length=100, fitness_fn=fitness_fn, maximize=True)

    rhc_curve, rhc_time = time_fn(
        partial(
            mlrose_hiive.random_hill_climb,
            problem,
            **dict(
                max_attempts=MAX_ATTEMPTS,
                max_iters=MAX_ITERS,
                curve=True,
                random_state=24,
                restarts=10,
            )
        )
    )
    sa_curve, sa_time = time_fn(
        partial(
            mlrose_hiive.simulated_annealing,
            problem,
            **dict(
                max_attempts=MAX_ATTEMPTS,
                max_iters=MAX_ITERS,
                curve=True,
                random_state=24,
                schedule=mlrose_hiive.GeomDecay(init_temp=2, decay=0.8, min_temp=0.001),
            )
        )
    )
    ga_curve, ga_time = time_fn(
        partial(
            mlrose_hiive.genetic_alg,
            problem,
            **dict(
                max_attempts=MAX_ATTEMPTS,
                max_iters=MAX_ITERS,
                curve=True,
                random_state=24,
                pop_size=200,
                mutation_prob=0.2,
            )
        )
    )
    mimic_curve, mimic_time = time_fn(
        partial(
            mlrose_hiive.mimic,
            problem,
            **dict(
                max_attempts=MAX_ATTEMPTS,
                max_iters=MAX_ITERS,
                curve=True,
                random_state=24,
                keep_pct=0.25,
            )
        )
    )

    x = range(1, 101)
    plt.plot(x, rhc_curve, label="RHC")
    plt.plot(x, sa_curve, label="SA")
    plt.plot(x, ga_curve, label="GA")
    plt.plot(x, mimic_curve, label="MIMIC")
    plt.legend()
    plt.title("Fitness - Queens problem")
    plt.xlabel("Iterations")
    plt.ylabel("Fitness")
    plt.savefig("plots/queens_fitness.png")
    plt.close()

    print("Queens Runtimes")
    print(f"Hill-Climbing: {rhc_time}")
    print(f"Simulated Annealing: {sa_time}")
    print(f"Genetic Algorithm: {ga_time}")
    print(f"MIMIC: {mimic_time}\n")


def six_peaks():
    fitness_fn = mlrose_hiive.SixPeaks(t_pct=0.16)
    problem = mlrose_hiive.DiscreteOpt(length=100, fitness_fn=fitness_fn, maximize=True)

    rhc_curve, rhc_time = time_fn(
        partial(
            mlrose_hiive.random_hill_climb,
            problem,
            **dict(
                max_attempts=MAX_ATTEMPTS,
                max_iters=MAX_ITERS,
                curve=True,
                random_state=24,
                restarts=100,
            )
        )
    )
    sa_curve, sa_time = time_fn(
        partial(
            mlrose_hiive.simulated_annealing,
            problem,
            **dict(
                max_attempts=MAX_ATTEMPTS,
                max_iters=MAX_ITERS,
                curve=True,
                random_state=24,
                schedule=mlrose_hiive.GeomDecay(init_temp=1, decay=0.1, min_temp=.1),
            )
        )
    )
    ga_curve, ga_time = time_fn(
        partial(
            mlrose_hiive.genetic_alg,
            problem,
            **dict(
                max_attempts=MAX_ATTEMPTS,
                max_iters=MAX_ITERS,
                curve=True,
                random_state=24,
                pop_size=200,
            )
        )
    )
    mimic_curve, mimic_time = time_fn(
        partial(
            mlrose_hiive.mimic,
            problem,
            **dict(
                max_attempts=MAX_ATTEMPTS,
                max_iters=MAX_ITERS,
                curve=True,
                random_state=24,
            )
        )
    )

    x = range(1, 101)
    plt.plot(x, rhc_curve, label="RHC")
    plt.plot(x, sa_curve, label="SA")
    plt.plot(x, ga_curve, label="GA")
    plt.plot(x, mimic_curve, label="MIMIC")
    plt.legend()
    plt.title("Fitness - Six Peaks problem")
    plt.xlabel("Iterations")
    plt.ylabel("Fitness")
    plt.savefig("plots/sixpeaks_fitness.png")
    plt.close()

    print("Six Peaks Runtimes")
    print(f"Hill-Climbing: {rhc_time}")
    print(f"Simulated Annealing: {sa_time}")
    print(f"Genetic Algorithm: {ga_time}")
    print(f"MIMIC: {mimic_time}\n")


def k_colors():
    G = nx.gnm_random_graph(72, 192, seed=np.random)
    fitness_fn = mlrose_hiive.MaxKColor(G.edges())
    colors = [random.random() for i in range(len(G.nodes))]
    nx.draw(G, node_color=colors)
    plt.savefig("plots/k-colors-graph.png")
    plt.close()

    problem = mlrose_hiive.DiscreteOpt(length=75, fitness_fn=fitness_fn, maximize=True)

    rhc_curve, rhc_time = time_fn(
        partial(
            mlrose_hiive.random_hill_climb,
            problem,
            **dict(
                max_attempts=MAX_ATTEMPTS,
                max_iters=MAX_ITERS,
                curve=True,
                random_state=24,
            )
        )
    )
    sa_curve, sa_time = time_fn(
        partial(
            mlrose_hiive.simulated_annealing,
            problem,
            **dict(
                max_attempts=MAX_ATTEMPTS,
                max_iters=MAX_ITERS,
                curve=True,
                random_state=24,
                schedule=mlrose_hiive.GeomDecay(init_temp=32, decay=0.8, min_temp=0.001),
            )
        )
    )
    ga_curve, ga_time = time_fn(
        partial(
            mlrose_hiive.genetic_alg,
            problem,
            **dict(
                max_attempts=MAX_ATTEMPTS,
                max_iters=MAX_ITERS,
                curve=True,
                random_state=24,
            )
        )
    )
    mimic_curve, mimic_time = time_fn(
        partial(
            mlrose_hiive.mimic,
            problem,
            **dict(
                max_attempts=MAX_ATTEMPTS,
                max_iters=MAX_ITERS,
                curve=True,
                pop_size=500,
                keep_pct=0.1,
                random_state=24,
            )
        )
    )

    x = range(1, 101)
    plt.plot(x, rhc_curve, label="RHC")
    plt.plot(x, sa_curve, label="SA")
    plt.plot(x, ga_curve, label="GA")
    plt.plot(x, mimic_curve, label="MIMIC")
    plt.legend()
    plt.title("Fitness - K Colors problem")
    plt.xlabel("Iterations")
    plt.ylabel("Fitness")
    plt.savefig("plots/kcolors_fitness.png")
    plt.close()

    print("K Colors Runtimes")
    print(f"Hill-Climbing: {rhc_time}")
    print(f"Simulated Annealing: {sa_time}")
    print(f"Genetic Algorithm: {ga_time}")
    print(f"MIMIC: {mimic_time}\n")


def create_datasets():
    n_samples = [732, 16144]
    n_features = [12, 18]
    n_redundant = [1, 3]
    datasets = []
    for samples, features, redundant in zip(
        n_samples,
        n_features,
        n_redundant,
    ):
        x, y = make_classification(
            n_samples=samples,
            n_classes=2,
            n_features=features,
            n_redundant=redundant,
            n_informative=features - redundant,
            n_clusters_per_class=1,
            random_state=1560152,  # from p1
        )
        datasets.append((x, y))

    return datasets


def nn_sgd():
    datasets = create_datasets()
    dataset1, _ = datasets
    X, y = dataset1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    nn = mlrose_hiive.NeuralNetwork(
        hidden_nodes=[600],
        algorithm="gradient_descent",
        max_iters=5000,
        bias=True,
        learning_rate=0.0001,
        restarts=0,
        curve=True,
        random_state=24,
    )

    start_time = time.time()
    nn.fit(X_train, y_train)
    end_time = time.time()
    print(f"Gradient Descent NN Runtime: {end_time - start_time}")
    y_pred = nn.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    metrics("gd", test_acc, "test")
    y_pred = nn.predict(X_train)
    train_acc = accuracy_score(y_train, y_pred)
    metrics("gd", train_acc, "train")

    plt.plot(nn.fitness_curve)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.savefig("plots/nn_sgd.png")
    plt.close()


def nn_rhc():
    datasets = create_datasets()
    dataset1, _ = datasets
    X, y = dataset1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    nn = mlrose_hiive.NeuralNetwork(
        hidden_nodes=[600],
        algorithm="random_hill_climb",
        early_stopping=True,
        max_attempts=100,
        max_iters=5000,
        bias=True,
        learning_rate=0.0001,
        restarts=0,
        curve=True,
        random_state=24,
    )

    start_time = time.time()
    nn.fit(X_train, y_train)
    end_time = time.time()
    print(f"Random Hill Climbing NN Runtime: {end_time - start_time}")
    y_pred = nn.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    metrics("rhc", test_acc, "test")
    y_pred = nn.predict(X_train)
    train_acc = accuracy_score(y_train, y_pred)
    metrics("rhc", train_acc, "train")

    plt.plot(nn.fitness_curve)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.savefig("plots/nn_rhc.png")
    plt.close()


def nn_sa():
    datasets = create_datasets()
    dataset1, _ = datasets
    X, y = dataset1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    nn = mlrose_hiive.NeuralNetwork(
        hidden_nodes=[600],
        algorithm="simulated_annealing",
        early_stopping=True,
        max_attempts=100,
        max_iters=5000,
        bias=True,
        learning_rate=0.0001,
        restarts=0,
        curve=True,
        random_state=24,
    )
    start_time = time.time()
    nn.fit(X_train, y_train)
    end_time = time.time()
    print(f"Simulated Annealing NN Runtime: {end_time - start_time}")
    y_pred = nn.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    metrics("sa", test_acc, "test")
    y_pred = nn.predict(X_train)
    train_acc = accuracy_score(y_train, y_pred)
    metrics("sa", train_acc, "train")

    plt.plot(nn.fitness_curve)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.savefig("plots/nn_sa.png")
    plt.close()


def nn_ga():
    datasets = create_datasets()
    dataset1, _ = datasets
    X, y = dataset1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    nn = mlrose_hiive.NeuralNetwork(
        hidden_nodes=[600],
        algorithm="genetic_alg",
        early_stopping=True,
        max_attempts=100,
        max_iters=5000,
        bias=True,
        learning_rate=0.0001,
        restarts=0,
        curve=True,
        random_state=24,
    )

    start_time = time.time()
    nn.fit(X_train, y_train)
    end_time = time.time()
    print(f"Genetic Algorithm NN Runtime: {end_time - start_time}")
    y_pred = nn.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    metrics("ga", test_acc, "test")
    y_pred = nn.predict(X_train)
    train_acc = accuracy_score(y_train, y_pred)
    metrics("ga", train_acc, "train")

    plt.plot(nn.fitness_curve)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.savefig("plots/nn_ga.png")
    plt.close()


def main():
    start_time = time.time()
    with multiprocessing.Pool(processes=multiprocessing.cpu_count() // 2) as pool:
        for future in [pool.apply_async(t) for t in tqdm([
            k_colors,
            queens,
            six_peaks,
            nn_sgd,
            nn_rhc,
            nn_sa,
            nn_ga,
        ])]:
            future.get()

    end_time = time.time()
    print(f"Total Runtime: {end_time - start_time}")


if __name__ == "__main__":
    fire.Fire(main)
