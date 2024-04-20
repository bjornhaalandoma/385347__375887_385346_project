from sklearn.model_selection import train_test_split
from src import utils
import argparse

import numpy as np

from src.data import load_data
from src.methods.dummy_methods import DummyClassifier
from src.methods.logistic_regression import LogisticRegression
from src.methods.linear_regression import LinearRegression
from src.methods.knn import KNN
from src.utils import normalize_fn, append_bias_term, accuracy_fn, macrof1_fn, mse_fn
import matplotlib.pyplot as plt

import os
np.random.seed(100)


def main(args):
    """
    The main function of the script. Do not hesitate to play with it
    and add your own code, visualization, prints, etc!

    Arguments:
        args (Namespace): arguments that were parsed from the command line (see at the end
                          of this file). Their value can be accessed as "args.argument".
    """
    # 1. First, we load our data and flatten the images into vectors

    # EXTRACTED FEATURES DATASET
    if args.data_type == "features":
        feature_data = np.load('features.npz', allow_pickle=True)
        xtrain, xtest, ytrain, ytest, ctrain, ctest = feature_data['xtrain'], feature_data['xtest'], \
            feature_data['ytrain'], feature_data['ytest'], feature_data['ctrain'], feature_data['ctest']

    # ORIGINAL IMAGE DATASET (MS2)
    elif args.data_type == "original":
        data_dir = os.path.join(args.data_path, 'dog-small-64')
        xtrain, xtest, ytrain, ytest, ctrain, ctest = load_data(data_dir)

    # TODO: ctrain and ctest are for regression task. (To be used for Linear Regression and KNN)
    # TODO: xtrain, xtest, ytrain, ytest are for classification task. (To be used for Logistic Regression and KNN)

    # 2. Then we must prepare it. This is were you can create a validation set,
    #  normalize, add bias, etc.

    # Make a validation set (it can overwrite xtest, ytest)
    if not args.test:
        xtrain_mean = np.mean(xtrain, axis=0)
        xtrain_std = np.std(xtrain, axis=0)

        normalized_xtrain = normalize_fn(xtrain, xtrain_mean, xtrain_std)
        normalized_xtest = normalize_fn(xtest, xtrain_mean, xtrain_std)

        x_train_with_bias = append_bias_term(normalized_xtrain)
        x_test_with_bias = append_bias_term(normalized_xtest)

        x_train_sub, xval, ytrain_sub, yval = train_test_split(
            xtrain, ytrain, test_size=0.3, random_state=42)

        normalized_x_train_sub = normalize_fn(
            x_train_sub, xtrain_mean, xtrain_std)
        normalized_xval = normalize_fn(xval, xtrain_mean, xtrain_std)

        x_train_sub_with_bias = append_bias_term(normalized_x_train_sub)
        xval_with_bias = append_bias_term(normalized_xval)

    # 3. Initialize the method you want to use.

    # Use NN (FOR MS2!)
    if args.method == "nn":
        raise NotImplementedError("This will be useful for MS2.")

    # Follow the "DummyClassifier" example for your methods
    if args.method == "dummy_classifier":
        method_obj = DummyClassifier(arg1=1, arg2=2)

    elif args.method == "logistic_regression":
        method_obj = LogisticRegression(lr=args.lr, max_iters=args.max_iters)

    elif args.method == "linear_regression":
        method_obj = LinearRegression(lmda=args.lmda)

    elif args.method == "knn":
        if args.task == "center_locating":
            method_obj = KNN(k=args.K, task_kind="regression")
        else:
            method_obj = KNN(k=args.K)

    # 4. Train and evaluate the method

    # TODO Checking if ctrain is correct

    if args.task == "center_locating":

        if args.plot_k:
            k_values = []
            losses = []
            for i in range(1, 21):
                model = KNN(k=i, task_kind="regression")
                model.fit(append_bias_term(xtrain), ctrain)
                preds = model.predict(append_bias_term(xtest))
                loss = mse_fn(preds, ctest)
                print(f"K={i} - Test set: loss = {loss:.3f}")
                k_values.append(i)
                losses.append(loss)

            fig, ax = plt.subplots()
            ax.plot(k_values, losses, 'ro-')
            ax.set_xlabel('K')
            ax.set_ylabel('Loss')
            ax.set_xticks(k_values)
            ax.set_title('Loss for different K values')
            plt.tight_layout()
            plt.show()

        else:

            preds_train = method_obj.fit(append_bias_term(xtrain), ctrain)

            # Perform inference for training and test data
            train_pred = method_obj.predict(append_bias_term(xtrain))
            preds = method_obj.predict(append_bias_term(xtest))

            # Report results: performance on train and valid/test sets
            train_loss = mse_fn(train_pred, ctrain)
            loss = mse_fn(preds, ctest)

            print(f"\nTrain loss = {train_loss:.3f}% - Test loss = {loss:.3f}")

    elif args.task == "breed_identifying":

        if args.plot_k:
            k_values = []
            accuracies = []
            for i in range(1, 21):  # Assuming you want to plot k from 1 to 10
                model = KNN(k=i)
                model.fit(x_train_with_bias, ytrain)
                preds = model.predict(x_test_with_bias)
                acc = accuracy_fn(preds, ytest)
                print(f"K={i} - Test set: accuracy = {acc:.3f}%")

        # Save the k value and its corresponding accuracy
                k_values.append(i)
                accuracies.append(acc)

    # Now plot all accuracies after the loop
            fig, ax = plt.subplots()
            # 'ro-' for red circles with line2
            ax.plot(k_values, accuracies, 'ro-')
            ax.set_xlabel('K')
            ax.set_ylabel('Accuracy (%)')
            ax.set_xticks(k_values)  # Set x-ticks to be each k value
            ax.set_title('Accuracy for different K values')
            plt.tight_layout()
            plt.show()

        if args.plot_hyperparameters:

            lambda_reg_values = [100, 10, 1, 0.1, 0.001]
            learning_rates = [0.0001, 0.001, 0.005, 0.01, 0.1]
            max_iterations_range = range(5, 500, 20)

            for lambda_reg in lambda_reg_values:
                results = {}

                # Initialize results storage for this lambda_reg value
                for lr in learning_rates:
                    results[lr] = {'max_iters': [], 'accuracies': []}

                # Run the training and validation loop
                for lr in learning_rates:
                    for max_iters in max_iterations_range:
                        model = LogisticRegression(
                            lr=lr, max_iters=max_iters, lambda_reg=lambda_reg)
                        model.fit(x_train_sub_with_bias, ytrain_sub)

                        val_acc = accuracy_fn(
                            model.predict(xval_with_bias), yval)
                        results[lr]['max_iters'].append(max_iters)
                        results[lr]['accuracies'].append(val_acc)

                # Find the best hyperparameter and its accuracy for this lambda_reg
                best_lr = max(results, key=lambda lr: max(
                    results[lr]['accuracies']))
                best_acc = max(max(results[lr]['accuracies'])
                               for lr in learning_rates)
                best_iters = max_iterations_range[results[best_lr]['accuracies'].index(
                    best_acc)]

                # Plot the results for this lambda_reg
                fig, ax = plt.subplots()
                for lr in results:
                    ax.plot(results[lr]['max_iters'], results[lr]
                            ['accuracies'], label=f'lr={lr}')

                best_point, = ax.plot(best_iters, best_acc, 'ro')
                ax.annotate(f'{best_acc:.2f}%', xy=(best_iters, best_acc), xytext=(8, 0),
                            textcoords='offset points', ha='center', va='center',
                            color="white", bbox=dict(boxstyle='round,pad=0.2', fc='red', alpha=0.5))

                labels = [f'lr={lr}' for lr in learning_rates]
                labels.append(
                    f'Best Accuracy: {best_acc:.2f}% (lr={best_lr}, iters={best_iters}, lambda={lambda_reg})')
                handles, _ = ax.get_legend_handles_labels()
                handles.append(best_point)

                ax.set_xlabel('Maximum Iterations')
                ax.set_ylabel('Accuracy')
                ax.set_title(f'Hyperparameter Tuning for lambda={lambda_reg}')
                ax.legend(handles=handles, labels=labels, loc='best')

                plt.tight_layout()
                plt.show()

        else:

            # Fit (:=train) the method on the training data for classification task. Append bias term.
            preds_train = method_obj.fit(x_train_with_bias, ytrain)

            # Predict on unseen data. Append bias term.
            preds = method_obj.predict(x_test_with_bias)

            # Report results: performance on train and valid/test sets
            acc = accuracy_fn(preds_train, ytrain)
            macrof1 = macrof1_fn(preds_train, ytrain)
            print(
                f"\nTrain set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

            acc = accuracy_fn(preds, ytest)
            macrof1 = macrof1_fn(preds, ytest)
            print(
                f"Test set:  accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

    else:
        raise Exception(
            "Invalid choice of task! Only support center_locating and breed_identifying!")

    # WRITE YOUR CODE HERE if you want to add other outputs, visualization, etc.


if __name__ == '__main__':
    # Definition of the arguments that can be given through the command line (terminal).
    # If an argument is not given, it will take its default value as defined below.
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default="center_locating",
                        type=str, help="center_locating / breed_identifying")
    parser.add_argument('--method', default="dummy_classifier", type=str,
                        help="dummy_classifier / knn / linear_regression/ logistic_regression / nn (MS2)")
    parser.add_argument('--data_path', default="data",
                        type=str, help="path to your dataset")
    parser.add_argument('--data_type', default="features",
                        type=str, help="features/original(MS2)")
    parser.add_argument('--lmda', type=float, default=10,
                        help="lambda of linear/ridge regression")
    parser.add_argument('--K', type=int, default=1,
                        help="number of neighboring datapoints used for knn")
    parser.add_argument('--lr', type=float, default=1e-5,
                        help="learning rate for methods with learning rate")
    parser.add_argument('--max_iters', type=int, default=100,
                        help="max iters for methods which are iterative")
    parser.add_argument('--test', action="store_true",
                        help="train on whole training data and evaluate on the test data, otherwise use a validation set")
    parser.add_argument('--plot_hyperparameters', action='store_true',
                        help="If set, plots the hyperparameter tuning results")
    parser.add_argument('--plot_k', action='store_true',
                        help="If set, plots the hyperparameter tuning results")

    # Feel free to add more arguments here if you need!

    # MS2 arguments
    parser.add_argument('--nn_type', default="cnn",
                        help="which network to use, can be 'Transformer' or 'cnn'")
    parser.add_argument('--nn_batch_size', type=int,
                        default=64, help="batch size for NN training")

    # "args" will keep in memory the arguments and their values,
    # which can be accessed as "args.data", for example.
    args = parser.parse_args()
    main(args)
