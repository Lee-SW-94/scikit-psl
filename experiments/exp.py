import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from skpsl.estimators import ProbabilisticScoringList, GeneticProbabilisticScoringList
from skpsl.preprocessing import MinEntropyBinarizer

if __name__ == '__main__':
    df = pd.read_csv("../data/player_binary.csv", index_col=0)
    X = df.iloc[:].values
    y = df.iloc[:].index.values

    #df = pd.read_csv("../data/41945.csv", header=None, index_col=None)
    #X = df.iloc[:, :-1]
    #y = df.iloc[:, -1]

    #df = pd.read_csv("../data/42900.csv", header=None, index_col=None)
    #X = df.iloc[:, :-1]
    #y = df.iloc[:, -1]

    X = MinEntropyBinarizer().fit_transform(X, y)
    classes_ = np.unique(y)
    y_ = np.array(y == classes_[1], dtype=int)

    time_fitness = []
    brier = []

    for i in range(10):
        X_train, X_test, y_train, y_test = train_test_split(X, y_, test_size=.2)

        psl = ProbabilisticScoringList({-3, -2, -1, 1, 2, 3})
        psl.fit(X_train, y_train)
        time_fitness.append(['GS', i+1, 1, 0, psl.fitness(X_train, y_train), psl.time])
        for j in range(len(psl.stage_clfs)):
            brier.append(['GS', i+1, 1, j, psl.score(X_test, y_test, k=j)])

        features = np.array(psl.stage_clfs[-1].features, dtype=int)
        scores = np.array(psl.stage_clfs[-1].scores, dtype=int)
        scores_sorted = [None]*len(features)
        for j in range(len(features)):
            scores_sorted[features[j]] = scores[j]
        result = list(features) + list(scores_sorted)

        for j in range(10):
            ga_psl1 = GeneticProbabilisticScoringList({-3, -2, -1, 1, 2, 3})
            ga_psl1.fit_ox4(X_train, y_train, given_solution=None)
            for k in range(len(ga_psl1.data)):
                time_fitness.append(['GA_OX4', i + 1, j + 1] + ga_psl1.data[k])

            for k in range(len(ga_psl1.stage_clfs)):
                brier.append(['GA_OX4', i + 1, j + 1, k, ga_psl1.score(X_test, y_test, k=k)])

        for j in range(10):
            ga_psl2 = GeneticProbabilisticScoringList({-3, -2, -1, 1, 2, 3})
            ga_psl2.fit_ox(X_train, y_train, given_solution=None)
            for k in range(len(ga_psl2.data)):
                time_fitness.append(['GA_OX', i + 1, j + 1] + ga_psl2.data[k])

            for k in range(len(ga_psl2.stage_clfs)):
                brier.append(['GA_OX', i + 1, j + 1, k, ga_psl2.score(X_test, y_test, k=k)])


    tf = pd.DataFrame(time_fitness, columns=['model', 'split', 'iteration', 'generation', 'fitness', 'time'])
    tf.to_csv("comparison_time_fit.csv")

    bs = pd.DataFrame(brier, columns=['model', 'split', 'iteration', 'stage', 'score'])
    bs.to_csv("comparison_brier.csv")