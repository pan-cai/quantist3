# -*- coding: utf-8 -*-

# quantist
# 
# Copyright 2017-2018 Pan Liu
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
# by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Unless required

""" 
Author: liupan 
"""

"""
Description:
"""

import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_gaussian_quantiles
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import matplotlib.pyplot as plt

from sklearn import ensemble
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn.ensemble.partial_dependence import partial_dependence
from sklearn.datasets.california_housing import fetch_california_housing
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn import clone
from sklearn.datasets import load_iris
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              AdaBoostClassifier)
from sklearn.tree import DecisionTreeClassifier
from itertools import product

import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier

from sklearn import datasets
from sklearn.model_selection import cross_val_predict
from sklearn import linear_model
import matplotlib.pyplot as plt

class SimpleML(object):
    def ml_kmeans(self):
        data_path = "../data/pool/"
        result_path = "../data/result/"
        data = pd.read_excel(data_path + "sh2.xls")
        # p_cahnge = data['p_change']
        p_cahnge = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
        kmeas = KMeans(n_clusters=2, random_state=0).fit(p_cahnge)
        kmeas.predict([[0, 0], [4, 4]])
        print(kmeas.cluster_centers_)
        print()

    def ml_pca(self):
        X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
        pca = PCA(n_components=2).fit(X)
        print(pca.explained_variance_ratio_)
        print(pca.singular_values_)
        print("---------------")
        pca2 = PCA(n_components=2, svd_solver='full').fit(
            X)  # n_components must be between 0 and n_features=2 with svd_solver='full'
        print(pca2.explained_variance_ratio_)
        print(pca2.singular_values_)
        print("---------------")
        pca3 = PCA(n_components=1, svd_solver='arpack').fit(X)
        print(pca3.explained_variance_ratio_)
        print(pca3.singular_values_)

    def ml_AdaBoostClassifile(self):
        # Demonstrate Gradient Boosting on the Boston housing dataset.

        # Construct dataset
        X1, y1 = make_gaussian_quantiles(cov=2.,
                                         n_samples=200, n_features=2,
                                         n_classes=2, random_state=1)
        X2, y2 = make_gaussian_quantiles(mean=(3, 3), cov=1.5,
                                         n_samples=300, n_features=2,
                                         n_classes=2, random_state=1)
        X = np.concatenate((X1, X2))
        y = np.concatenate((y1, - y2 + 1))

        # Create and fit an AdaBoosted decision tree
        bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                                 algorithm="SAMME",
                                 n_estimators=200)

        bdt.fit(X, y)

        plot_colors = "br"
        plot_step = 0.02
        class_names = "AB"

        plt.figure(figsize=(10, 5))

        # Plot the decision boundaries
        plt.subplot(121)
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                             np.arange(y_min, y_max, plot_step))

        Z = bdt.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
        plt.axis("tight")

        # Plot the training points
        for i, n, c in zip(range(2), class_names, plot_colors):
            idx = np.where(y == i)
            plt.scatter(X[idx, 0], X[idx, 1],
                        c=c, cmap=plt.cm.Paired,
                        s=20, edgecolor='k',
                        label="Class %s" % n)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.legend(loc='upper right')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Decision Boundary')

        # Plot the two-class decision scores
        twoclass_output = bdt.decision_function(X)
        plot_range = (twoclass_output.min(), twoclass_output.max())
        plt.subplot(122)
        for i, n, c in zip(range(2), class_names, plot_colors):
            plt.hist(twoclass_output[y == i],
                     bins=10,
                     range=plot_range,
                     facecolor=c,
                     label='Class %s' % n,
                     alpha=.5,
                     edgecolor='k')
        x1, x2, y1, y2 = plt.axis()
        plt.axis((x1, x2, y1, y2 * 1.2))
        plt.legend(loc='upper right')
        plt.ylabel('Samples')
        plt.xlabel('Score')
        plt.title('Decision Scores')

        plt.tight_layout()
        plt.subplots_adjust(wspace=0.35)
        plt.show()

    def ml_AdaBoostRegressor(self):
        # Create the dataset
        rng = np.random.RandomState(1)
        X = np.linspace(0, 6, 100)[:, np.newaxis]
        y = np.sin(X).ravel() + np.sin(6 * X).ravel() + rng.normal(0, 0.1, X.shape[0])

        # Fit regression model
        regr_1 = DecisionTreeRegressor(max_depth=4)

        regr_2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),
                                   n_estimators=300, random_state=rng)

        regr_1.fit(X, y)
        regr_2.fit(X, y)

        # Predict
        y_1 = regr_1.predict(X)
        y_2 = regr_2.predict(X)

        # Plot the results
        plt.figure()
        plt.scatter(X, y, c="k", label="training samples")
        plt.plot(X, y_1, c="g", label="n_estimators=1", linewidth=2)
        plt.plot(X, y_2, c="r", label="n_estimators=300", linewidth=2)
        plt.xlabel("data")
        plt.ylabel("target")
        plt.title("Boosted Decision Tree Regression")
        plt.legend()
        plt.show()

    def ml_BaggingRegressor(self):
        # Settings
        n_repeat = 50  # Number of iterations for computing expectations
        n_train = 50  # Size of the training set
        n_test = 1000  # Size of the test set
        noise = 0.1  # Standard deviation of the noise
        np.random.seed(0)

        # Change this for exploring the bias-variance decomposition of other
        # estimators. This should work well for estimators with high variance (e.g.,
        # decision trees or KNN), but poorly for estimators with low variance (e.g.,
        # linear models).
        estimators = [("Tree", DecisionTreeRegressor()),
                      ("Bagging(Tree)", BaggingRegressor(DecisionTreeRegressor()))]

        n_estimators = len(estimators)

        # Generate data
        def f(x):
            x = x.ravel()

            return np.exp(-x ** 2) + 1.5 * np.exp(-(x - 2) ** 2)

        def generate(n_samples, noise, n_repeat=1):
            X = np.random.rand(n_samples) * 10 - 5
            X = np.sort(X)

            if n_repeat == 1:
                y = f(X) + np.random.normal(0.0, noise, n_samples)
            else:
                y = np.zeros((n_samples, n_repeat))

                for i in range(n_repeat):
                    y[:, i] = f(X) + np.random.normal(0.0, noise, n_samples)

            X = X.reshape((n_samples, 1))

            return X, y

        X_train = []
        y_train = []

        for i in range(n_repeat):
            X, y = generate(n_samples=n_train, noise=noise)
            X_train.append(X)
            y_train.append(y)

        X_test, y_test = generate(n_samples=n_test, noise=noise, n_repeat=n_repeat)

        plt.figure(figsize=(10, 8))

        # Loop over estimators to compare
        for n, (name, estimator) in enumerate(estimators):
            # Compute predictions
            y_predict = np.zeros((n_test, n_repeat))

            for i in range(n_repeat):
                estimator.fit(X_train[i], y_train[i])
                y_predict[:, i] = estimator.predict(X_test)

            # Bias^2 + Variance + Noise decomposition of the mean squared error
            y_error = np.zeros(n_test)

            for i in range(n_repeat):
                for j in range(n_repeat):
                    y_error += (y_test[:, j] - y_predict[:, i]) ** 2

            y_error /= (n_repeat * n_repeat)

            y_noise = np.var(y_test, axis=1)
            y_bias = (f(X_test) - np.mean(y_predict, axis=1)) ** 2
            y_var = np.var(y_predict, axis=1)

            print("{0}: {1:.4f} (error) = {2:.4f} (bias^2) "
                  " + {3:.4f} (var) + {4:.4f} (noise)".format(name,
                                                              np.mean(y_error),
                                                              np.mean(y_bias),
                                                              np.mean(y_var),
                                                              np.mean(y_noise)))

            # Plot figures
            plt.subplot(2, n_estimators, n + 1)
            plt.plot(X_test, f(X_test), "b", label="$f(x)$")
            plt.plot(X_train[0], y_train[0], ".b", label="LS ~ $y = f(x)+noise$")

            for i in range(n_repeat):
                if i == 0:
                    plt.plot(X_test, y_predict[:, i], "r", label="$\^y(x)$")
                else:
                    plt.plot(X_test, y_predict[:, i], "r", alpha=0.05)

            plt.plot(X_test, np.mean(y_predict, axis=1), "c",
                     label="$\mathbb{E}_{LS} \^y(x)$")

            plt.xlim([-5, 5])
            plt.title(name)

            if n == n_estimators - 1:
                plt.legend(loc=(1.1, .5))

            plt.subplot(2, n_estimators, n_estimators + n + 1)
            plt.plot(X_test, y_error, "r", label="$error(x)$")
            plt.plot(X_test, y_bias, "b", label="$bias^2(x)$"),
            plt.plot(X_test, y_var, "g", label="$variance(x)$"),
            plt.plot(X_test, y_noise, "c", label="$noise(x)$")

            plt.xlim([-5, 5])
            plt.ylim([0, 0.1])

            if n == n_estimators - 1:
                plt.legend(loc=(1.1, .5))

        plt.subplots_adjust(right=.75)
        plt.show()

    def ml_GradientBoostingClassifier(self):
        return None

    def ml_GradientBoostingClassifier(self):

        # #############################################################################
        # Load data
        boston = datasets.load_boston()
        X, y = shuffle(boston.data, boston.target, random_state=13)
        X = X.astype(np.float32)
        offset = int(X.shape[0] * 0.9)
        X_train, y_train = X[:offset], y[:offset]
        X_test, y_test = X[offset:], y[offset:]

        # #############################################################################
        # Fit regression model
        params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
                  'learning_rate': 0.01, 'loss': 'ls'}
        clf = ensemble.GradientBoostingRegressor(**params)

        clf.fit(X_train, y_train)
        mse = mean_squared_error(y_test, clf.predict(X_test))
        print("MSE: %.4f" % mse)

        # #############################################################################
        # Plot training deviance

        # compute test set deviance
        test_score = np.zeros((params['n_estimators'],), dtype=np.float64)

        for i, y_pred in enumerate(clf.staged_predict(X_test)):
            test_score[i] = clf.loss_(y_test, y_pred)

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.title('Deviance')
        plt.plot(np.arange(params['n_estimators']) + 1, clf.train_score_, 'b-',
                 label='Training Set Deviance')
        plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
                 label='Test Set Deviance')
        plt.legend(loc='upper right')
        plt.xlabel('Boosting Iterations')
        plt.ylabel('Deviance')

        # #############################################################################
        # Plot feature importance
        feature_importance = clf.feature_importances_
        # make importances relative to max importance
        feature_importance = 100.0 * (feature_importance / feature_importance.max())
        sorted_idx = np.argsort(feature_importance)
        pos = np.arange(sorted_idx.shape[0]) + .5
        plt.subplot(1, 2, 2)
        plt.barh(pos, feature_importance[sorted_idx], align='center')
        plt.yticks(pos, boston.feature_names[sorted_idx])
        plt.xlabel('Relative Importance')
        plt.title('Variable Importance')
        plt.show()

    def ml_GradientBoostingClassifier2(self):
        # This example shows how to obtain partial dependence plots from a GradientBoostingRegressor trained on the California housing dataset.
        cal_housing = fetch_california_housing()

        # split 80/20 train-test
        X_train, X_test, y_train, y_test = train_test_split(cal_housing.data,
                                                            cal_housing.target,
                                                            test_size=0.2,
                                                            random_state=1)
        names = cal_housing.feature_names

        print("Training GBRT...")
        clf = GradientBoostingRegressor(n_estimators=100, max_depth=4,
                                        learning_rate=0.1, loss='huber',
                                        random_state=1)
        clf.fit(X_train, y_train)
        print(" done.")

        print('Convenience plot with ``partial_dependence_plots``')

        features = [0, 5, 1, 2, (5, 1)]
        fig, axs = plot_partial_dependence(clf, X_train, features,
                                           feature_names=names,
                                           n_jobs=3, grid_resolution=50)
        fig.suptitle('Partial dependence of house value on nonlocation features\n'
                     'for the California housing dataset')
        plt.subplots_adjust(top=0.9)  # tight_layout causes overlap with suptitle

        print('Custom 3d plot via ``partial_dependence``')
        fig = plt.figure()

        target_feature = (1, 5)
        pdp, axes = partial_dependence(clf, target_feature,
                                       X=X_train, grid_resolution=50)
        XX, YY = np.meshgrid(axes[0], axes[1])
        Z = pdp[0].reshape(list(map(np.size, axes))).T
        ax = Axes3D(fig)
        surf = ax.plot_surface(XX, YY, Z, rstride=1, cstride=1,
                               cmap=plt.cm.BuPu, edgecolor='k')
        ax.set_xlabel(names[target_feature[0]])
        ax.set_ylabel(names[target_feature[1]])
        ax.set_zlabel('Partial dependence')
        #  pretty init view
        ax.view_init(elev=22, azim=122)
        plt.colorbar(surf)
        plt.suptitle('Partial dependence of house value on median\n'
                     'age and average occupancy')
        plt.subplots_adjust(top=0.9)

        plt.show()

    def ml_RandomForestClassifier(self):
        #Plot the class probabilities of the first sample in a toy dataset predicted by three different classifiers and averaged by the VotingClassifier.


        clf1 = LogisticRegression(random_state=123)
        clf2 = RandomForestClassifier(random_state=123)
        clf3 = GaussianNB()
        X = np.array([[-1.0, -1.0], [-1.2, -1.4], [-3.4, -2.2], [1.1, 1.2]])
        y = np.array([1, 1, 2, 2])

        eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)],
                                voting='soft',
                                weights=[1, 1, 5])

        # predict class probabilities for all classifiers
        probas = [c.fit(X, y).predict_proba(X) for c in (clf1, clf2, clf3, eclf)]

        # get class probabilities for the first sample in the dataset
        class1_1 = [pr[0, 0] for pr in probas]
        class2_1 = [pr[0, 1] for pr in probas]

        # plotting

        N = 4  # number of groups
        ind = np.arange(N)  # group positions
        width = 0.35  # bar width

        fig, ax = plt.subplots()

        # bars for classifier 1-3
        p1 = ax.bar(ind, np.hstack(([class1_1[:-1], [0]])), width,
                    color='green', edgecolor='k')
        p2 = ax.bar(ind + width, np.hstack(([class2_1[:-1], [0]])), width,
                    color='lightgreen', edgecolor='k')

        # bars for VotingClassifier
        p3 = ax.bar(ind, [0, 0, 0, class1_1[-1]], width,
                    color='blue', edgecolor='k')
        p4 = ax.bar(ind + width, [0, 0, 0, class2_1[-1]], width,
                    color='steelblue', edgecolor='k')

        # plot annotations
        plt.axvline(2.8, color='k', linestyle='dashed')
        ax.set_xticks(ind + width)
        ax.set_xticklabels(['LogisticRegression\nweight 1',
                            'GaussianNB\nweight 1',
                            'RandomForestClassifier\nweight 5',
                            'VotingClassifier\n(average probabilities)'],
                           rotation=40,
                           ha='right')
        plt.ylim([0, 1])
        plt.title('Class probabilities for sample 1 by different classifiers')
        plt.legend([p1[0], p2[0]], ['class 1', 'class 2'], loc='upper left')
        plt.show()

    def ml_RandomForestClassifier2(self):
        #Plot the decision surfaces of forests of randomized trees trained on pairs of features of the iris dataset.


        # Parameters
        n_classes = 3
        n_estimators = 30
        cmap = plt.cm.RdYlBu
        plot_step = 0.02  # fine step width for decision surface contours
        plot_step_coarser = 0.5  # step widths for coarse classifier guesses
        RANDOM_SEED = 13  # fix the seed on each iteration

        # Load data
        iris = load_iris()

        plot_idx = 1

        models = [DecisionTreeClassifier(max_depth=None),
                  RandomForestClassifier(n_estimators=n_estimators),
                  ExtraTreesClassifier(n_estimators=n_estimators),
                  AdaBoostClassifier(DecisionTreeClassifier(max_depth=3),
                                     n_estimators=n_estimators)]

        for pair in ([0, 1], [0, 2], [2, 3]):
            for model in models:
                # We only take the two corresponding features
                X = iris.data[:, pair]
                y = iris.target

                # Shuffle
                idx = np.arange(X.shape[0])
                np.random.seed(RANDOM_SEED)
                np.random.shuffle(idx)
                X = X[idx]
                y = y[idx]

                # Standardize
                mean = X.mean(axis=0)
                std = X.std(axis=0)
                X = (X - mean) / std

                # Train
                clf = clone(model)
                clf = model.fit(X, y)

                scores = clf.score(X, y)
                # Create a title for each column and the console by using str() and
                # slicing away useless parts of the string
                model_title = str(type(model)).split(
                    ".")[-1][:-2][:-len("Classifier")]

                model_details = model_title
                if hasattr(model, "estimators_"):
                    model_details += " with {} estimators".format(
                        len(model.estimators_))
                print(model_details + " with features", pair,
                      "has a score of", scores)

                plt.subplot(3, 4, plot_idx)
                if plot_idx <= len(models):
                    # Add a title at the top of each column
                    plt.title(model_title)

                # Now plot the decision boundary using a fine mesh as input to a
                # filled contour plot
                x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
                y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
                xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                                     np.arange(y_min, y_max, plot_step))

                # Plot either a single DecisionTreeClassifier or alpha blend the
                # decision surfaces of the ensemble of classifiers
                if isinstance(model, DecisionTreeClassifier):
                    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
                    Z = Z.reshape(xx.shape)
                    cs = plt.contourf(xx, yy, Z, cmap=cmap)
                else:
                    # Choose alpha blend level with respect to the number
                    # of estimators
                    # that are in use (noting that AdaBoost can use fewer estimators
                    # than its maximum if it achieves a good enough fit early on)
                    estimator_alpha = 1.0 / len(model.estimators_)
                    for tree in model.estimators_:
                        Z = tree.predict(np.c_[xx.ravel(), yy.ravel()])
                        Z = Z.reshape(xx.shape)
                        cs = plt.contourf(xx, yy, Z, alpha=estimator_alpha, cmap=cmap)

                # Build a coarser grid to plot a set of ensemble classifications
                # to show how these are different to what we see in the decision
                # surfaces. These points are regularly space and do not have a
                # black outline
                xx_coarser, yy_coarser = np.meshgrid(
                    np.arange(x_min, x_max, plot_step_coarser),
                    np.arange(y_min, y_max, plot_step_coarser))
                Z_points_coarser = model.predict(np.c_[xx_coarser.ravel(),
                                                       yy_coarser.ravel()]
                                                 ).reshape(xx_coarser.shape)
                cs_points = plt.scatter(xx_coarser, yy_coarser, s=15,
                                        c=Z_points_coarser, cmap=cmap,
                                        edgecolors="none")

                # Plot the training points, these are clustered together and have a
                # black outline
                plt.scatter(X[:, 0], X[:, 1], c=y,
                            cmap=ListedColormap(['r', 'y', 'b']),
                            edgecolor='k', s=20)
                plot_idx += 1  # move on to the next plot in sequence

        plt.suptitle("Classifiers on feature subsets of the Iris dataset")
        plt.axis("tight")

        plt.show()

    def ml_VotingClassifier(self):


        # Loading some example data
        iris = datasets.load_iris()
        X = iris.data[:, [0, 2]]
        y = iris.target

        # Training classifiers
        clf1 = DecisionTreeClassifier(max_depth=4)
        clf2 = KNeighborsClassifier(n_neighbors=7)
        clf3 = SVC(kernel='rbf', probability=True)
        eclf = VotingClassifier(estimators=[('dt', clf1), ('knn', clf2),
                                            ('svc', clf3)],
                                voting='soft', weights=[2, 1, 2])

        clf1.fit(X, y)
        clf2.fit(X, y)
        clf3.fit(X, y)
        eclf.fit(X, y)

        # Plotting decision regions
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                             np.arange(y_min, y_max, 0.1))

        f, axarr = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(10, 8))

        for idx, clf, tt in zip(product([0, 1], [0, 1]),
                                [clf1, clf2, clf3, eclf],
                                ['Decision Tree (depth=4)', 'KNN (k=7)',
                                 'Kernel SVM', 'Soft Voting']):
            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)

            axarr[idx[0], idx[1]].contourf(xx, yy, Z, alpha=0.4)
            axarr[idx[0], idx[1]].scatter(X[:, 0], X[:, 1], c=y,
                                          s=20, edgecolor='k')
            axarr[idx[0], idx[1]].set_title(tt)

        plt.show()

    def ml_LinearRegression(self):
        #cross_val_predict
        lr = linear_model.LinearRegression()
        boston = datasets.load_boston()
        y = boston.target

        # cross_val_predict returns an array of the same size as `y` where each entry
        # is a prediction obtained by cross validation:
        predicted = cross_val_predict(lr, boston.data, y, cv=10)

        fig, ax = plt.subplots()
        ax.scatter(y, predicted, edgecolors=(0, 0, 0))
        ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
        ax.set_xlabel('Measured')
        ax.set_ylabel('Predicted')
        plt.show()



# KMeas
s = SimpleML()
# s.ml_kmeans()
# s.ml_pca()
# s.ml_AdaBoostClassifile()
#s.ml_AdaBoostRegressor()
s.ml_LinearRegression()