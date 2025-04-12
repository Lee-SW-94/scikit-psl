import time
from typing import Optional

from numpy import random, ndarray
import numpy as np
import pygad
import pandas as pd

from itertools import permutations

from sklearn.exceptions import NotFittedError
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, brier_score_loss


from skpsl.estimators import ProbabilisticScoringList, ProbabilisticScoringSystem
from skpsl.preprocessing import MinEntropyBinarizer

class GeneticProbabilisticScoringList(ProbabilisticScoringList):

    def __init__(
            self,
            score_set: set,
            loss_cutoff: float = None,
            method="bisect",
            lookahead=1,
            n_jobs=None,
            stage_loss=None,
            cascade_loss=None,
            stage_clf_params=None,
    ):
        super().__init__(score_set, loss_cutoff, method, lookahead, n_jobs, stage_loss, cascade_loss, stage_clf_params)
        self.calibrator = None
        self.ga_instance = None
        self.data = []


    def fit_ox4(
        self,
        X,
        y,
        given_solution = None,
        sample_weight=None,
        predef_features: Optional[np.ndarray] = None,
        predef_scores: Optional[np.ndarray] = None,
        strict=True
    ) -> "ProbabilisticScoringList":
        start_time = time.time()

        number_features = int(X.shape[1]) # Number of Features
        X = np.array(X)

        self.classes_ = np.unique(y)
        y_ = np.array(y == self.classes_[1], dtype=int)

        feature_order = list(range(number_features))
        scores = list(self.score_set)
        scores.sort()

        self.calibrator = IsotonicRegression(
            y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip"
        )

        data = []

        def initialize(sol_per_pop=10):
            init=[]
            if given_solution is not None:
                sol_per_pop = sol_per_pop - 1
                init.append(given_solution)

            for _ in range(sol_per_pop):
                init_feature_order = list(np.random.permutation(feature_order))
                init_scores = list(np.random.choice(scores, size=number_features, replace=True))

                init.append(init_feature_order+init_scores)

            return init

        def fitness(ga_instance, solution, solution_idx):
            loss_value = []
            order = np.array(solution[:number_features], dtype=int)
            score = np.array(solution[number_features:], dtype=int)

            sorted_data = X[:, order]
            sorted_scores = list(score[order])

            for i in range(number_features + 1):
                total = GeneticProbabilisticScoringList.compute_total_score(sorted_data[:,:i], sorted_scores[:i])
                if i == 0:
                    total = np.zeros((X.shape[0], 1))
                self.calibrator = self.calibrator.fit(total, y_)

                y_prob = self.predict_prob(total)

                loss_value.append(log_loss(y_, y_prob[:, 1]))
                #loss_value.append(brier_score_loss(y_, y_prob[:, 1]))
            return 1/sum(loss_value)

        def ox4(parents, offspring_size, ga_instance):
            off=[]

            for a, b in permutations(range(parents.shape[0]),2):
                order1, score1 = parents[a, :number_features].copy(), parents[a, number_features:].copy()
                order2, score2 = parents[b, :number_features].copy(), parents[b, number_features:].copy()


                # order-crossover (ox4)
                random_split_point1 = np.random.choice(range(1, number_features), size=2, replace=False)
                random_split_point1.sort()
                random_split_point2 = np.random.choice(range(1, number_features), size=2, replace=False)
                random_split_point2.sort()

                off_order1 = [None]*number_features
                off_order2 = [None]*number_features

                inherit1 = list(order1[random_split_point1[0]:random_split_point1[1]])
                temp_order2 = list(order2[random_split_point2[1]:])+list(order2[:random_split_point2[1]])
                remaining_elements1 = [x for x in temp_order2 if x not in inherit1]

                off_order1[random_split_point1[0]:random_split_point1[1]] = inherit1
                off_order1[random_split_point1[1]:] = remaining_elements1[:number_features - random_split_point1[1]]
                off_order1[:random_split_point1[0]] = remaining_elements1[number_features - random_split_point1[1]:]


                inherit2 = list(order2[random_split_point2[0]:random_split_point2[1]])
                temp_order1 = list(order1[random_split_point1[1]:])+list(order1[:random_split_point1[1]])
                remaining_elements2 = [x for x in temp_order1 if x not in inherit2]

                off_order2[random_split_point2[0]:random_split_point2[1]] = inherit2
                off_order2[random_split_point2[1]:] = remaining_elements2[:number_features - random_split_point2[1]]
                off_order2[:random_split_point2[0]] = remaining_elements2[number_features - random_split_point2[1]:]


                # score-crossover
                beta = random.uniform(low=-0.25, high=1.25)

                off_score1=((score1*beta) + (score2*(1-beta))).astype(float)
                off_score1=ndarray.round(off_score1,0).astype(int)
                for i in range(number_features):
                    if off_score1[i] not in scores:
                        if off_score1[i] < scores[0]:
                            off_score1[i] = scores[0]
                        elif off_score1[i] > scores[-1]:
                            off_score1[i] = scores[-1]
                        elif scores[0] < off_score1[i] < scores[-1]:
                            idx_min = np.argmin(abs(scores - off_score1[i]))
                            off_score1[i] = scores[idx_min]

                off_score2=((score2*beta) + (score1*(1-beta))).astype(float)
                off_score2=ndarray.round(off_score2,0).astype(int)
                for i in range(number_features):
                    if off_score2[i] not in scores:
                        if off_score2[i]<scores[0]:
                            off_score2[i] = scores[0]
                        elif off_score2[i]>scores[-1]:
                            off_score2[i] = scores[-1]
                        elif scores[0]<off_score2[i]<scores[-1]:
                            idx_min = np.argmin(abs(scores-off_score1[i]))
                            off_score2[i] = scores[idx_min]


                assert (off_order1[i] in range(number_features) for i in range(number_features)) and len(np.unique(off_order1)) == number_features
                assert (off_score1[i] in scores for i in range(number_features)) and len(off_score1) == number_features
                assert (off_order2[i] in range(number_features) for i in range(number_features)) and len(np.unique(off_order2)) == number_features
                assert (off_score2[i] in scores for i in range(number_features)) and len(off_score2) == number_features

                off_score1 = list(off_score1)
                off_score2 = list(off_score2)
                off.append(off_order1+off_score1)
                off.append(off_order2+off_score2)

            rand = np.random.choice(range(len(off)), size=offspring_size[0], replace=False)
            offspring = [off[i] for i in rand]
            return np.array(offspring)

        def mutate(offspring, ga_instance):
            #swap 2 elements
            order = offspring[:, :number_features]
            score = offspring[:, number_features:]

            mutated = []
            for i in range(offspring.shape[0]):
                off_order = order[i]
                off_score = score[i]

                if random.random() < ga_instance.mutation_probability:
                    # order mutate - swap
                    swap_idx = np.random.choice(range(number_features), size=2, replace=False)
                    off_order[swap_idx[0]], off_order[swap_idx[1]] = off_order[swap_idx[1]], off_order[swap_idx[0]]

                    # score mutate - swap
                    swap_idx = np.random.choice(range(number_features), size=2, replace=False)
                    off_score[swap_idx[0]], off_score[swap_idx[1]] = off_score[swap_idx[1]], off_score[swap_idx[0]]

                mutated.append(list(off_order)+list(off_score))
            return np.array(mutated)

        def on_generation(ga_instance):
            _, fit, _ = ga_instance.best_solution()
            current_time = time.time()-start_time
            self.data.append([ga_instance.generations_completed, fit, current_time])


        self.ga_instance = pygad.GA(
            num_generations=200,
            num_parents_mating=5,
            fitness_func=fitness,
            initial_population=initialize(),
            gene_space=feature_order * number_features + scores * number_features,
            gene_type=[int] * number_features + [int] * number_features,
            num_genes=2*number_features,
            parent_selection_type="rws",
            keep_elitism=2,
            crossover_type=ox4,
            mutation_type=mutate,
            mutation_probability=0.05,
            on_generation=on_generation,
        )

        self.ga_instance.run()
        solution, fitness, _ = self.ga_instance.best_solution()
        self.ga_instance.plot_fitness()

        sol_order = np.array(solution[:number_features], dtype=int)
        sol_score = np.array(solution[number_features:], dtype=int)

        self.stage_clfs = []
        for i in range(number_features + 1):
            order_i = sol_order[:i]

            sorted_scores = sol_score[order_i]
            score_i = sorted_scores[:i]

            k_clf = ProbabilisticScoringSystem(
                features=list(order_i),
                scores=list(score_i),
                initial_feature_thresholds=None,
                **self.stage_clf_params_,
            ).fit(X, y)
            self.stage_clfs.append(k_clf)

        return self

    def fit_ox(
        self,
        X,
        y,
        given_solution = None,
        sample_weight=None,
        predef_features: Optional[np.ndarray] = None,
        predef_scores: Optional[np.ndarray] = None,
        strict=True
    ) -> "ProbabilisticScoringList":
        start_time = time.time()

        number_features = int(X.shape[1]) # Number of Features
        X = np.array(X)

        self.classes_ = np.unique(y)
        y_ = np.array(y == self.classes_[1], dtype=int)

        feature_order = list(range(number_features))
        scores = list(self.score_set)
        scores.sort()

        self.calibrator = IsotonicRegression(
            y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip"
        )

        data = []

        def initialize(sol_per_pop=10):
            init=[]
            if given_solution is not None:
                sol_per_pop = sol_per_pop - 1
                init.append(given_solution)

            for _ in range(sol_per_pop):
                init_feature_order = list(np.random.permutation(feature_order))
                init_scores = list(np.random.choice(scores, size=number_features, replace=True))

                init.append(init_feature_order+init_scores)

            return init

        def fitness(ga_instance, solution, solution_idx):
            loss_value = []
            order = np.array(solution[:number_features], dtype=int)
            score = np.array(solution[number_features:], dtype=int)

            sorted_data = X[:, order]
            sorted_scores = list(score[order])

            for i in range(number_features + 1):
                total = GeneticProbabilisticScoringList.compute_total_score(sorted_data[:,:i], sorted_scores[:i])
                if i == 0:
                    total = np.zeros((X.shape[0], 1))
                self.calibrator = self.calibrator.fit(total, y_)

                y_prob = self.predict_prob(total)

                loss_value.append(log_loss(y_, y_prob[:, 1]))
                #loss_value.append(brier_score_loss(y_, y_prob[:, 1]))
            return 1/sum(loss_value)

        def ox(parents, offspring_size, ga_instance):
            off=[]

            for a, b in permutations(range(parents.shape[0]),2):
                order1, score1 = parents[a, :number_features].copy(), parents[a, number_features:].copy()
                order2, score2 = parents[b, :number_features].copy(), parents[b, number_features:].copy()


                # order-crossover (ox)
                random_split_point = np.random.choice(range(1, number_features), size=2, replace=False)
                random_split_point.sort()


                off_order1 = [None]*number_features
                off_order2 = [None]*number_features

                inherit1 = list(order1[random_split_point[0]:random_split_point[1]])
                remaining_elements1 = [x for x in order2 if x not in inherit1]

                off_order1[random_split_point[0]:random_split_point[1]] = inherit1
                off_order1[:random_split_point[0]] = remaining_elements1[:random_split_point[0]]
                off_order1[random_split_point[1]:] = remaining_elements1[random_split_point[0]:]

                inherit2 = list(order2[random_split_point[0]:random_split_point[1]])
                remaining_elements2 = [x for x in order1 if x not in inherit2]

                off_order2[random_split_point[0]:random_split_point[1]] = inherit2
                off_order2[:random_split_point[0]] = remaining_elements2[:random_split_point[0]]
                off_order2[random_split_point[1]:] = remaining_elements2[random_split_point[0]:]


                # score-crossover (single point crossover)
                cross_point1 = np.random.choice(range(1, number_features))
                cross_point2 = np.random.choice(range(1, number_features))

                off_score1 = list(score1[:cross_point1]) + list(score2[cross_point1:])
                off_score2 = list(score2[:cross_point2]) + list(score1[cross_point2:])

                assert (off_order1[i] in range(number_features) for i in range(number_features)) and len(np.unique(off_order1)) == number_features
                assert (off_score1[i] in scores for i in range(number_features)) and len(off_score1) == number_features
                assert (off_order2[i] in range(number_features) for i in range(number_features)) and len(np.unique(off_order2)) == number_features
                assert (off_score2[i] in scores for i in range(number_features)) and len(off_score2) == number_features

                off.append(list(off_order1) + list(off_score1))
                off.append(list(off_order2) + list(off_score2))

            rand = np.random.choice(range(len(off)), size=offspring_size[0], replace=False)
            offspring = [off[i] for i in rand]
            return np.array(offspring)

        def mutate(offspring, ga_instance):
            order = offspring[:, :number_features]
            score = offspring[:, number_features:]

            mutated = []
            for i in range(offspring.shape[0]):
                off_order = order[i]
                off_score = score[i]

                if random.random() < ga_instance.mutation_probability:
                    # order mutate - swap
                    swap_idx = np.random.choice(range(number_features), size=2, replace=False)
                    off_order[swap_idx[0]], off_order[swap_idx[1]] = off_order[swap_idx[1]], off_order[swap_idx[0]]

                    # score mutate - replace
                    mut_index = np.random.choice(range(number_features))
                    off_score[mut_index] = np.random.choice(scores)

                mutated.append(list(off_order) + list(off_score))
            return np.array(mutated)

        def on_generation(ga_instance):
            _, fit, _ = ga_instance.best_solution()
            current_time = time.time()-start_time
            self.data.append([ga_instance.generations_completed, fit, current_time])

        self.ga_instance = pygad.GA(
            num_generations=200,
            num_parents_mating=5,
            fitness_func=fitness,
            initial_population=initialize(),
            gene_space=feature_order * number_features + scores * number_features,
            gene_type=[int] * number_features + [int] * number_features,
            num_genes=2 * number_features,
            parent_selection_type="sss",
            keep_elitism=2,
            crossover_type=ox,
            mutation_type=mutate,
            mutation_probability=0.1,
            on_generation=on_generation,
        )

        self.ga_instance.run()
        solution, fitness, _ = self.ga_instance.best_solution()
        self.ga_instance.plot_fitness()

        sol_order = np.array(solution[:number_features], dtype=int)
        sol_score = np.array(solution[number_features:], dtype=int)

        self.stage_clfs = []
        for i in range(number_features + 1):
            order_i = sol_order[:i]

            sorted_scores = sol_score[order_i]
            score_i = sorted_scores[:i]

            k_clf = ProbabilisticScoringSystem(
                features=list(order_i),
                scores=list(score_i),
                initial_feature_thresholds=None,
                **self.stage_clf_params_,
            ).fit(X, y)
            self.stage_clfs.append(k_clf)

        return self

    @staticmethod
    def compute_total_score(data, score):
        data_ = np.vstack(data)
        scores_ = np.array(score)

        return (data_ @ scores_).reshape(-1, 1)

    def predict_prob(self, total):
        sigma, idx = np.unique(total, return_inverse=True)

        ps = self.calibrator.transform(sigma.reshape(-1, 1))
        proba_pos = ps[idx]

        return np.vstack([1 - proba_pos, proba_pos]).T

    def score(self, X, y, k=None, sample_weight=None):
        """
        Calculates the Brier score of the model
        :param X:
        :param y:
        :param k: Classifier stage to use for prediction
        :param sample_weight: ignored
        :return:
        """
        if self.stage_clfs is None:
            raise NotFittedError()

        y_ = np.array(y == self.classes_[1], dtype=int)

        if k is None:
            brier = []
            for i in range(len(self.stage_clfs)):
                order = np.array(self.stage_clfs[i].features, dtype=int)
                scores = np.array(self.stage_clfs[i].scores, dtype=int)

                sorted_data = X[:, order]
                total = GeneticProbabilisticScoringList.compute_total_score(sorted_data,scores)
                if len(order) == 0:
                    total = np.zeros((X.shape[0], 1))
                self.calibrator = self.stage_clfs[i].calibrator
                y_prob = self.predict_prob(total)
                brier.append(brier_score_loss(y_, y_prob[:, 1]))
            return sum(brier)

        order = np.array(self.stage_clfs[k].features, dtype=int)
        scores = np.array(self.stage_clfs[k].scores, dtype=int)

        sorted_data = X[:, order]

        total = GeneticProbabilisticScoringList.compute_total_score(sorted_data, scores)
        if len(order) == 0:
            total = np.zeros((X.shape[0], 1))
        self.calibrator = self.stage_clfs[k].calibrator
        y_prob = self.predict_prob(total)
        return brier_score_loss(y_, y_prob[:, 1])

    def inspect(self, k=None, feature_names=None) -> pd.DataFrame:
        """
        Inspect the scoring system, returning a DataFrame of active features, their scores.

        :param k: Number of stages to include in the output. If None, all stages are included.
        :param feature_names: Optional list of names for the features. If None, indices are used.
        :return: A DataFrame containing the features, scores, and thresholds of the scoring system.
        """
        if self.ga_instance is None:
            raise ValueError("The model must be fitted before calling inspect.")

        solution, _, _ =self.ga_instance.best_solution()
        num_features = int(len(solution) / 2)

        k = k or len(self) - 1

        scores = self.stage_clfs[k].scores
        features = self.stage_clfs[k].features

        all_total_scores = [{0}]
        for i, score in enumerate(scores):
            all_total_scores.append(
                all_total_scores[i]
                | {prev_sum + score for prev_sum in all_total_scores[i]}
            )

        data = []
        for clf, T in zip(self.stage_clfs[: k + 1], all_total_scores):
            a = {t: np.nan for t in all_total_scores[-1]}
            probas = clf.calibrator.transform(np.array(list(T)))
            a.update(dict(zip(T, probas)))
            data.append(a)

        df = pd.DataFrame(data)
        df = df[sorted(df.columns)]
        df.columns = [f"T = {t_}" for t_ in df.columns]
        df.insert(0, "Score", np.array([np.nan] + list(scores)))
        if feature_names is not None:
            if len(feature_names) != num_features:
                raise ValueError(
                    f"Passed feature names are of incorrect length! Passed {len(feature_names)}, expected {len(features)}."
                )
            feature_names = [feature_names[i] for i in features]
            df.insert(0, "Feature", [np.nan] + list(feature_names[:k]))
        else:
            df.insert(
                0,
                "Feature Index",
                [np.nan] + list(features[:k]),
            )

        return df.reset_index(names=["Stage"])


if __name__ == '__main__':
    #from sklearn.datasets import load_breast_cancer

    #data = load_breast_cancer()
    #X, y = data.data, data.target
    #X = MinEntropyBinarizer().fit_transform(X, y)

    #ga_psl = GeneticProbabilisticScoringList({-1, 1, 2})
    #ga_psl.fit(X, y)

    from sklearn.model_selection import train_test_split
    df = pd.read_csv("../../data/player_binary.csv", index_col=0).sample(200)
    X = df.iloc[:, :-1].values
    y = df.iloc[:].index.values
    X = MinEntropyBinarizer().fit_transform(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

    ga_psl = GeneticProbabilisticScoringList({-2, -1, 1, 2})
    ga_psl.fit_ox(X_train, y_train)

    ins = ga_psl.inspect()
    print(ins.to_string(index=False, na_rep="-", justify="center", float_format=lambda x: f"{x:.2f}"))