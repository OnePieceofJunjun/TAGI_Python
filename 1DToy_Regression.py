import json, sys
from numpy.random import seed
import time
import numpy as np
import torch
from Models.meanfield import Bayesian_Network_torch as Bayesian_Network
from helper_functions.aux_function import *
from helper_functions.plots import plot_1DToy_plot
seed(1)
import tensorflow as tf
tf.random.set_seed(2)
tf.get_logger().setLevel('ERROR')
# max_epochs = 50
# patience = 10
# best_epoch = 0
# best_ll_val = -np.Inf
# wait = 0
# LL_val = []
# LL_obs = []
# Early Stopping parameters
# epochs = 50
# patience = 5
# best_nll_val =  np.Inf
# # float('inf')
# wait = 0# patience counter
# best_epoch = 0

def run_regression(ds, size, epochs, trials, hidden):
    #early stopping paramters
    epochs = 50
    patience = 5
    best_nll_val = np.Inf
    # float('inf')
    wait = 0  # patience counter
    best_epoch = 0
    # Judgement of the generalization  parameters
    improvement_window = 10
    val_nll_update=[]
    obs_nll_update=[]
    val_improvements=[]
    train_improvements=[]
    min_improvement = 0.0001



    t = time.strftime("%Y%m%d-%H%M%S")
    dir_name = f"./experiments/{ds}/exp_{t}"
    createFolder(dir_name)

    print("Dataset: ", ds)
    config_and_hyperparams_dict = {"size":size, "hidden neurons":hidden, "epochs":epochs, "trials":trials}

    X_train, y_train, X_train_scaler, y_train_scaler = create_data(dataset=ds, feature_range=[-1,1], data_size=size)
    X_test, y_test, X_test_scaler, y_test_scaler = create_data(dataset=ds, feature_range=[-1,1], data_size=int(size*0.1))
    input_dim = X_train.shape[1]
    output_dim = 1

    hidden_activation = ["relu2" for i in range(len(hidden))]
    output_activation = ["linear"]
    activations = hidden_activation + output_activation
    layers = [input_dim] + hidden + [output_dim]

    config_and_hyperparams_dict["activations"] = activations

    rmse_tagi = np.zeros(trials)
    nll_tagi = np.zeros(trials)
    train_times_tagi = np.zeros(trials)



    for t in range(trials):
        print(f"Trial {t+1}")
        bnn = Bayesian_Network(layers, activations, load_from_keras=False, input_scaler=X_train_scaler, output_scaler=y_train_scaler, verbose=False)

        # X_train, y_train = torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float()
        # X_test, y_test = torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float()
        X_train, y_train = torch.tensor(X_train).float(), torch.tensor(y_train).float()
        X_test, y_test = torch.tensor(X_test).float(), torch.tensor(y_test).float()

        print("Train TAGI")

        start = time.time()
        for e in range(epochs):
            bnn.train(X_train, y_train)
            # Prediction
            y_pred_obs, y_cov_obs = bnn.predict(X_train)
            y_pred_val, y_cov_val = bnn.predict(X_test)

            #calculate nll
            obs_rmse, obs_nll = rmse_regression(y_train, y_pred_obs, y_cov_obs)
            val_rmse, val_nll = rmse_regression(y_test, y_pred_val, y_cov_val)
            val_nll_update.append(val_nll)
            obs_nll_update.append(obs_nll)
         # Calculate improvements
         #    if e > 1:
         #        val_improvements_update = val_improvements.append(val_improvements(e))
         #        train_improvements_update = train_improvements.append(train_improvements(e))
         #
         #        val_improvements_update = val_nll_update(e) - val_nll_update(e - 1);
         #        train_improvements_update= obs_nll_update(e) - obs_nll_update(e - 1);
         #    end
            if e > 0:
                val_improvement = val_nll_update[-1] - val_nll_update[-2]
                train_improvement = obs_nll_update[-1] - obs_nll_update[-2]
                val_improvements.append(val_improvement)
                train_improvements.append(train_improvement)


            #Check for improvement early stopping
            if val_nll < best_nll_val:
                best_nll_val = val_nll
                best_epoch = e
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print(f"Early stopping at epoch{e}")
                    break

            # # Judge generalization
            # if e >= improvement_window:
            #     mean_val_improvement = np.mean(val_improvements[-improvement_window:])
            #     mean_train_improvement = np.mean(train_improvements[-improvement_window:])
            #     print(f'Epoch={e}.')
            #     print(f'Best epoch based on Val_LL: {best_epoch}')
            #     if mean_train_improvement < min_improvement and mean_val_improvement < min_improvement:
            #         print('The model is underfitting.')
            #     elif mean_train_improvement > mean_val_improvement:
            #         print('The model is overfitting.')
            #     else:
            #         print('The model generalizes well.')
            # else:
            #     print('The model generalizes well.')
            # #Judge generalization
            # if e >= improvement_window:
            #     mean_val_improvement= np.mean(val_nll_update[-improvement_window:e])
            #     mean_train_improvement=np.mean(obs_nll_update[-improvement_window:e])


            end = time.time()
        train_times_tagi[t] = end - start
        # Judge generalization
        if e >= improvement_window:
            mean_val_improvement = np.mean(val_improvements[-improvement_window:])
            mean_train_improvement = np.mean(train_improvements[-improvement_window:])
            print(f'Epoch={e}.')
            print(f'Best epoch based on Val_LL: {best_epoch}')
            if mean_train_improvement < min_improvement and mean_val_improvement < min_improvement:
                print('The model is underfitting.')
            elif mean_train_improvement > mean_val_improvement:
                print('The model is overfitting.')
            else:
                print('The model generalizes well.')
        else:
            print('The model generalizes well.')

        print(f"Best epoch based on validation log-likelihood:{best_epoch}")
        y_pred, y_cov = bnn.predict(X_test)
        rmse_tagi[t], nll_tagi[t] = rmse_regression(y_test, y_pred, y_cov)

    nll_tagi_mean, nll_tagi_std = np.mean(nll_tagi), np.std(nll_tagi)
    rmse_tagi_mean, rmse_tagi_std = np.mean(rmse_tagi), np.std(rmse_tagi)
    time_tagi_mean, time_tagi_std = np.mean(train_times_tagi), np.std(train_times_tagi)

    pm = u"\u00B1"

    print(f"TAGI RMSE: {'%.3f'%rmse_tagi_mean} {pm} {'%.3f'%rmse_tagi_std}")
    print(f"TAGI NLL: {'%.3f'%nll_tagi_mean} {pm} {'%.3f'%nll_tagi_std}")
    print(f"TAGI total train time / s: {'%.3f'%time_tagi_mean} {pm} {'%.3f'%time_tagi_std}")

    with open(os.path.join(dir_name, "config_and_hyperparams.json"), "w") as f:
        json.dump(config_and_hyperparams_dict, f, indent=4, sort_keys=True)

    fun = lambda u: u ** 3

    # Plotting results
    plt.figure(figsize=(12, 6))

    # Plot NLLs
    plt.subplot(1, 2, 1)
    plt.plot(val_nll_update, 'r-o', label='Validation NLL')
    plt.plot(obs_nll_update, 'k-o', label='Training NLL')
    plt.xlabel('Epoch')
    plt.ylabel('Negative Log-Likelihood')
    plt.title('NLL over Epochs')  # Corrected from 'tle' to 'title'
    plt.legend()

    # Plot improvements
    plt.subplot(1, 2, 2)
    if len(val_improvements) > 0 and len(train_improvements) > 0:
        plt.plot(val_improvements, 'r-o', label='Validation Improvement')
        plt.plot(train_improvements, 'k-o', label='Training Improvement')
    plt.xlabel('Epoch')
    plt.ylabel('Improvement')
    plt.title('Improvements over Epochs')  # Corrected from 'tle' to 'title'
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(dir_name, "training_validation_plot.png"))
    plt.show()



    plot_1DToy_plot(bnn, X_train, y_train, fun, dir_name, x_scaler=X_train_scaler, y_scaler=y_train_scaler)
    eval = np.stack([ rmse_tagi, train_times_tagi]).T
    np.savetxt(os.path.join(dir_name, "results.csv"), eval, delimiter=',', header="rmse_tagi,train_time")

if __name__ == "__main__":

    ds = "dataset_1DToy_regression"
    size = 20
    epochs = 50
    hidden_neurons = [100]
    trials = 1

    run_regression(ds, size, epochs, trials, hidden_neurons)








