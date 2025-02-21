import os

# Set TensorFlow to run in deterministic mode for reproducibility
os.environ['TF_DETERMINISTIC_OPS'] = '1'
import numpy as np
import pandas as pd
import tensorflow as tf
import MatchMaker
import performance_metrics
import argparse
import random

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# --------------- Parse MatchMaker arguments --------------- #

parser = argparse.ArgumentParser(description='REQUEST REQUIRED PARAMETERS OF MatchMaker')

parser.add_argument('--comb-data-name', default='/cta/users/ebcandir/matchmaker/data/drugcomb2/DrugCombV2.csv',
                    help="Name of the drug combination data")

parser.add_argument('--cell_line-gex', default='/cta/users/ebcandir/matchmaker/data/drugcomb2/cell_line_gex_dc2.csv',
                    help="Name of the cell line gene expression data")

parser.add_argument('--drug1-chemicals', default='/cta/users/ebcandir/matchmaker/data/drugcomb2/dc2_drug1_chem_py.csv',
                    help="Name of the chemical features data for drug 1")

parser.add_argument('--drug2-chemicals', default='/cta/users/ebcandir/matchmaker/data/drugcomb2/dc2_drug2_chem_py.csv',
                    help="Name of the chemical features data for drug 2")

parser.add_argument('--gpu-devices', default='0', type=str,
                    help='gpu device ids for CUDA_VISIBLE_DEVICES')

parser.add_argument('--train-test-mode', default=1, type = int,
                    help="Test of train mode (0: test, 1: train)")

parser.add_argument('--train-ind', default='/cta/users/ebcandir/matchmaker/data/drugcomb2/train_set_lpo',
                    help="Data indices that will be used for training")

parser.add_argument('--val-ind', default='/cta/users/ebcandir/matchmaker/data/drugcomb2/val_set_lpo',
                    help="Data indices that will be used for validation")

parser.add_argument('--test-ind', default='/cta/users/ebcandir/matchmaker/data/drugcomb2/test_set_lpo',
                    help="Data indices that will be used for test")

parser.add_argument('--arch', default='/cta/users/ebcandir/matchmaker/data/drugcomb2/scripts/architecture.txt',
                    help="Architecute file to construct MatchMaker layers")

parser.add_argument('--gpu-support', default=True,
                    help='Use GPU support or not')

parser.add_argument('--saved-model-name', default="/cta/users/ebcandir/matchmaker/data/drugcomb2/lpo/overfit_lpo_matchmaker_saved.h5",
                    help='Model name to save weights')

parser.add_argument('--saved-path', default="/cta/users/ebcandir/matchmaker/data/drugcomb2/lpo/")

parser.add_argument('--drug-features', default=0, type = int,
                    help="Type of the drug features (0: drug feature, 1: one hot encoded feature)")

parser.add_argument('--cell-line-features', default=0, type = int,
                    help="Type of the cell line features (0: cell line feature, 1: one hot encoded feature)")
parser.add_argument('--count', default=0, type = int,
                    help="")
args = parser.parse_args()
# ---------------------------------------------------------- #
num_cores = 8
print(tf.config.list_physical_devices('GPU'))

# If there are GPUs available, set TensorFlow to use only the first GPU
if tf.config.list_physical_devices('GPU'):
    try:
        # Specify which GPU to use
        tf.config.set_visible_devices(tf.config.list_physical_devices('GPU')[0], 'GPU')
        # Configure TensorFlow to use only a specific amount of GPU memory
        for gpu in tf.config.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        # Catch runtime error if modification occurs after GPUs have been initialized
        print(e)

# load and process data
chem1, chem2, cell_line, synergies = MatchMaker.data_loader(args.drug1_chemicals, args.drug2_chemicals,
                                                args.cell_line_gex, args.comb_data_name, args.drug_features, args.cell_line_features)
# normalize and split data into train, validation and test
norm = 'tanh_norm'

i = args.count

# If there are GPUs available, set TensorFlow to use only the first GPU
if tf.config.list_physical_devices('GPU'):
    try:
        # Specify which GPU to use
        tf.config.set_visible_devices(tf.config.list_physical_devices('GPU')[0], 'GPU')
        # Configure TensorFlow to use only a specific amount of GPU memory
        for gpu in tf.config.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        # Catch runtime error if modification occurs after GPUs have been initialized
        print(e)
        
train_ind = f"{args.train_ind}_{i}.txt"
val_ind = f"{args.val_ind}_{i}.txt"
test_ind = f"{args.test_ind}_{i}.txt"

train_data, val_data, test_data = MatchMaker.prepare_data(chem1, chem2, cell_line, synergies, norm,
                                            train_ind, val_ind, test_ind, args.drug_features, args.cell_line_features)


# calculate weights for weighted MSE loss
min_s = np.amin(train_data['y'])
loss_weight = np.log(train_data['y'] - min_s + np.e)

# load architecture file
architecture = pd.read_csv(args.arch)

# prepare layers of the model and the model name
layers = {}
layers['DSN_1'] = architecture['DSN_1'][0] # layers of Drug Synergy Network 1
layers['DSN_2'] = architecture['DSN_2'][0] # layers of Drug Synergy Network 2
layers['SPN'] = architecture['SPN'][0] # layers of Synergy Prediction Network
modelName = f"{args.saved_model_name}_{i}.h5" # name of the model to save the weights

# define constant parameters of MatchMaker
l_rate = 0.0001
inDrop = 0.2
drop = 0.5
max_epoch = 1000
batch_size = 128
earlyStop_patience = 100

model = MatchMaker.generate_network(train_data, layers, inDrop, drop)

if (args.train_test_mode == 1):
    # if we are in training mode
    model = MatchMaker.trainer(model, l_rate, train_data, val_data, max_epoch, batch_size,
                                earlyStop_patience, modelName,loss_weight)
# load the best model
model.load_weights(modelName + ".keras")

# predict in Drug1, Drug2 order
pred1 = MatchMaker.predict(model, [test_data['drug1'],test_data['drug2']])
# predict in Drug2, Drug1 order
pred2 = MatchMaker.predict(model, [test_data['drug2'],test_data['drug1']])

# take the mean for final prediction
pred = (pred1 + pred2) / 2
mse_value = performance_metrics.mse(test_data['y'], pred)
spearman_value = performance_metrics.spearman(test_data['y'], pred)
pearson_value = performance_metrics.pearson(test_data['y'], pred)
se = performance_metrics.calculate_se(test_data['y'], pred)


# Save results for the current split
np.savetxt(f"{args.saved_path}y_test_{i}.txt", np.asarray(test_data['y']), delimiter=",")
np.savetxt(f"{args.saved_path}pred_{i}.txt", np.asarray(pred), delimiter=",")
print(f"Split {i}: MSE={mse_value}, SCC={spearman_value}, PCC={pearson_value}, SE={se}")



