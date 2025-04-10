import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from skpsl.estimators import ProbabilisticScoringList, GeneticProbabilisticScoringList
from skpsl.preprocessing import MinEntropyBinarizer

if __name__ == '__main__':
    df = pd.read_csv("../data/player_binary.csv", index_col=0)
    X = df.iloc[:,:-1].values
    y = df.iloc[:].index.values

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
            ga_psl = GeneticProbabilisticScoringList({-3, -2, -1, 1, 2, 3})
            ga_psl.fit(X_train, y_train, given_solution=result)
            for k in range(len(ga_psl.data)):
                time_fitness.append(['GA', i+1, j+1] + ga_psl.data[k])
            for k in range(len(ga_psl.stage_clfs)):
                brier.append(['GA', i+1, j+1, k, ga_psl.score(X_test, y_test, k=k)])


    tf = pd.DataFrame(time_fitness, columns=['model', 'split', 'iteration', 'generation', 'fitness', 'time'])
    tf.to_csv("pb_comparison_time_fit_2.csv")

    bs = pd.DataFrame(brier, columns=['model', 'split', 'iteration', 'stage', 'score'])
    bs.to_csv("pb_comparison_brier_2.csv")