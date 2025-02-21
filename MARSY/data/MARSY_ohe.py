import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Dropout, Activation
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr
from scipy import stats
from sklearn.metrics import mean_squared_error
import numpy as np

# Load main dataset
data = pd.read_csv('/cta/users/ebcandir/MARSY/data/output_ohe.csv').to_numpy()
data_targets = pd.read_csv('/cta/users/ebcandir/MARSY/data/data.csv')

targets = data_targets[['Ri1', 'Ri2', 'Synergy_Zip']].to_numpy()

"""
### Data ###
X_train = pd.read_csv('Sample_Train_Set.csv', delimiter = ',', index_col='Unnamed: 0').to_numpy()
Y_train = pd.read_csv('Sample_Input_Targets.csv', delimiter = ',', index_col='Unnamed: 0').to_numpy()

X_test = pd.read_csv('Sample_Test_Set.csv', delimiter = ',', index_col='Unnamed: 0').to_numpy()
Y_test = pd.read_csv('Predictions_Sample.csv', delimiter = ',', index_col='Unnamed: 0').to_numpy()
"""

# Function to load fold indices
def load_fold_indices(fold_num):
    train_indices = np.loadtxt(f'/cta/users/ebcandir/MARSY/data/lpo_folds/Pair_Tr{fold_num}.txt', dtype=int)
    test_indices = np.loadtxt(f'/cta/users/ebcandir/MARSY/data/lpo_folds/Pair_Tst{fold_num}.txt', dtype=int)
    return train_indices, test_indices

# Function to prepare data for the fold
def prepare_fold_data(data, targets, train_indices, test_indices):
    X_train, X_test = data[train_indices], data[test_indices]
    Y_train, Y_test = targets[train_indices], targets[test_indices]
    return X_train, X_test, Y_train, Y_test

### Data formatting to fit MARSY's input requirements ###
def data_preparation(X_tr, X_tst, pair_range):
    X_tr = []
    X_tst = []
    
    #Extract Pairs and Triples from the input vector of each sample
    #Pairs refers to features of both drugs (670*2 features)
    #Triple refers to the features of both drugs and the cancer cell line (670*2 + 75 features)
    pair = []
    for i in X_train:
        temp_pair = i[:pair_range]
        pair.append(temp_pair)
    pair = np.asarray(pair)
    x1 = np.asarray(X_train)
    
    X_tr.append(x1)
    X_tr.append(pair)
    
    pair = []
    for i in X_test:
        temp_pair = i[:pair_range]
        pair.append(temp_pair)
    pair = np.asarray(pair)
    x2 = np.asarray(X_test)
    
    X_tst.append(x2)
    X_tst.append(pair)
    
    return X_tr, X_tst    


### Implementation of the MARSY model ###
def MARSY(X_tr, Y_tr, param):
    #Encoder for Triple
    tuple_vec = Input(shape=(param[0]))
    tpl = Dense(2048, activation='linear', kernel_initializer='he_normal')(tuple_vec)
    tpl = Dropout(param[2])(tpl)
    out_tpl1 = Dense(4096, activation='relu')(tpl)
    model_tpl = Model(tuple_vec, out_tpl1)

    tpl_inp = Input(shape=(param[0]))
    out_tpl = model_tpl(tpl_inp)
    
    #Encoder for Pair
    pair_vec = Input(shape=(param[1]))
    pair1 = Dense(1024, activation='linear', kernel_initializer='he_normal')(pair_vec)
    pair1 = Dropout(param[2])(pair1)
    out_p1 = Dense(2048, activation = 'relu')(pair1)
    model_pair = Model(pair_vec, out_p1)

    pair_inp = Input(shape=(param[1]))
    out_pair = model_pair(pair_inp)

    #Decoder to predict the synergy score and the single drug response of each drug
    concatenated_tpl = keras.layers.concatenate([out_pair, out_tpl])
    out_c1 = Dense(4096, activation='relu')(concatenated_tpl)
    out_c1 = Dropout(param[3])(out_c1)
    out_c1 = Dense(1024, activation='relu')(out_c1)
    out_c1 = Dense(3, activation='linear', name="Predictor_Drug_Combination")(out_c1)

    multitask_model = Model(inputs= [tpl_inp, pair_inp], outputs =[out_c1])

    multitask_model.compile(optimizer= tf.keras.optimizers.Adamax(learning_rate=float(0.001), 
                                                    beta_1=0.9, beta_2=0.999, epsilon=1e-07), 
                                                    loss={'Predictor_Drug_Combination': 'mse'}, 
                                                    metrics={'Predictor_Drug_Combination': 'mse'})

    es = EarlyStopping(monitor='val_mse', mode='min', verbose=0, patience=param[6])

    multitask_model.fit(X_tr, Y_tr, batch_size=param[5], epochs=param[4], verbose=0, 
                                     validation_split=0.2, callbacks=[es])
    
    return multitask_model 


### Parameters ###
triple_length = 1415
pair_length = 1340
dropout_encoders = 0.2
dropout_decoder = 0.5
epochs = 200
batch_size = 64
tol_stopping = 10

param = [triple_length, pair_length, dropout_encoders, dropout_decoder, epochs, batch_size, tol_stopping]

num_folds = 5
### Training and Prediction Example ###
"""
training_set, testing_set = data_preparation(X_train, X_test, pair_length)
trained_MARSY = MARSY(training_set, Y_train, param)
pred = trained_MARSY.predict(testing_set)
"""

# Loop over each fold
scc_synergy, scc_rs1, scc_rs2 = [], [], []
pcc_synergy, pcc_rs1, pcc_rs2 = [], [], []
rmse_synergy, rmse_rs1, rmse_rs2 = [], [], []
mse_synergy, mse_rs1, mse_rs2 = [], [], []
se_synergy, se_rs1, se_rs2 = [], [], []

def pearson(y, pred):
    pear = stats.pearsonr(y, pred)
    pear_value = pear[0]
    pear_p_val = pear[1]
    print("Pearson correlation is {:.4f} and p-value is {:.4e}".format(pear_value, pear_p_val))
    return pear_value

def spearman(y, pred):
    spear = stats.spearmanr(y, pred)
    spear_value = spear[0]
    spear_p_val = spear[1]
    print("Spearman correlation is {:.4f} and p-value is {:.4e}".format(spear_value, spear_p_val))
    return spear_value

def mse(y, pred):
    err = mean_squared_error(y, pred)
    print("Mean squared error is {:.4f}".format(err))
    return err
def squared_error(y,pred):
    errs = []
    for i in range(y.shape[0]):
        err = (y[i]-pred[i]) * (y[i]-pred[i])
        errs.append(err)
    return np.asarray(errs)

# Function to calculate SE
def calculate_se(y, pred):
    # Calculate squared errors
    squared_errs = squared_error(y, pred)
    
    # Standard error of MSE
    mse_se = np.std(squared_errs, ddof=1) / np.sqrt(len(squared_errs))
    print("Standard Error (SE) of MSE is {}".format(mse_se))
    
    return mse_se

for fold_num in range(1, num_folds + 1):
    print(f"Training on Fold {fold_num}...")

    # Load the fold indices
    train_indices, test_indices = load_fold_indices(fold_num)

    # Prepare the data for this fold
    X_train, X_test, Y_train, Y_test = prepare_fold_data(data, targets, train_indices, test_indices)

    # Prepare data according to the model's input format
    training_set, testing_set = data_preparation(X_train, X_test, param[1])  # Pair length is param[1]

    # Train the MARSY model on the current fold's training data
    trained_MARSY = MARSY(training_set, Y_train, param)

    # Make predictions on the test set
    pred = trained_MARSY.predict(testing_set)
    
    Y_test_df = pd.DataFrame(Y_test, columns=["Truth_RS1", "Truth_RS2", "Truth_Zip"])
    Y_test_file = (f"/cta/users/ebcandir/MARSY/data/fold_{fold_num}_Y_test_ohe.csv")
    Y_test_df.to_csv(Y_test_file, index=False)
    
    pred_df = pd.DataFrame(pred, columns=["Pred_RS1", "Pred_RS2", "Pred_Zip"])
    pred_file = (f"/cta/users/ebcandir/MARSY/data/fold_{fold_num}_predictions_ohe.csv")
    pred_df.to_csv(pred_file, index=False)
    
    print(f"Predictions for Fold {fold_num} saved to {pred_file}")

    # Split predictions and true values into synergy score, rs1, and rs2
    pred_synergy, pred_rs1, pred_rs2 = pred[:, 2], pred[:, 0], pred[:, 1]
    y_synergy, y_rs1, y_rs2 = Y_test[:, 2], Y_test[:, 0], Y_test[:, 1]  # Synergy_Zip is in column 2, Ri1 (rs1) in 0, Ri2 (rs2) in 1

    # Calculate RMSE
    rmse_synergy.append(np.sqrt(mse(y_synergy, pred_synergy)))
    rmse_rs1.append(np.sqrt(mse(y_rs1, pred_rs1)))
    rmse_rs2.append(np.sqrt(mse(y_rs2, pred_rs2)))

    mse_synergy.append(mse(y_synergy, pred_synergy))
    mse_rs1.append(mse(y_rs1, pred_rs1))
    mse_rs2.append(mse(y_rs2, pred_rs2))

    se_synergy.append(calculate_se(y_synergy, pred_synergy))
    se_rs1.append(calculate_se(y_rs1, pred_rs1))
    se_rs2.append(calculate_se(y_rs2, pred_rs2))

    # Calculate PCC (Pearson's Correlation Coefficient)
    pcc_synergy.append(pearson(y_synergy, pred_synergy))  # We only need the correlation coefficient
    pcc_rs1.append(pearson(y_rs1, pred_rs1))
    pcc_rs2.append(pearson(y_rs2, pred_rs2))

    # Calculate SCC (Spearman's Correlation Coefficient)
    scc_synergy.append(spearman(y_synergy, pred_synergy))  # We only need the correlation coefficient
    scc_rs1.append(spearman(y_rs1, pred_rs1))
    scc_rs2.append(spearman(y_rs2, pred_rs2))

 
    print(f"Fold {fold_num} - RMSE (Synergy): {rmse_synergy[-1]}, MSE (Synergy): {mse_synergy[-1]}, SE (Synergy): {se_synergy[-1]}, PCC (Synergy): {pcc_synergy[-1]}, SCC (Synergy): {scc_synergy[-1]}")
    print(f"Fold {fold_num} - RMSE (RS1): {rmse_rs1[-1]}, MSE (RS1): {mse_rs1[-1]}, SE (RS1): {se_rs1[-1]}, PCC (RS1): {pcc_rs1[-1]}, SCC (RS1): {scc_rs1[-1]}")
    print(f"Fold {fold_num} - RMSE (RS2): {rmse_rs2[-1]}, MSE (RS2): {mse_rs2[-1]}, SE (RS2): {se_rs2[-1]}, PCC (RS2): {pcc_rs2[-1]}, SCC (RS2): {scc_rs2[-1]}")

# Calculate and print the average metrics across all folds
print(f"Average RMSE (Synergy): {np.mean(rmse_synergy)}, Average MSE (Synergy): {np.mean(mse_synergy)}, Average SE (Synergy): {np.mean(se_synergy)}, PCC (Synergy): {np.mean(pcc_synergy)}, SCC (Synergy): {np.mean(scc_synergy)}")
print(f"Average RMSE (RS1): {np.mean(rmse_rs1)}, Average MSE (RS1): {np.mean(mse_rs1)}, Average SE (RS1): {np.mean(se_rs1)}, PCC (RS1): {np.mean(pcc_rs1)}, SCC (RS1): {np.mean(scc_rs1)}")
print(f"Average RMSE (RS2): {np.mean(rmse_rs2)}, Average MSE (RS2): {np.mean(mse_rs2)}, Average SE (RS2): {np.mean(se_rs2)}, PCC (RS2): {np.mean(pcc_rs2)}, SCC (RS2): {np.mean(scc_rs2)}")
