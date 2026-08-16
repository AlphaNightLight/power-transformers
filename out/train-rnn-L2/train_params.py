import torch



# ##### #
# Torch #
# ##### #

args_gpu = 0
device = torch.device("cuda:" + str(args_gpu) if torch.cuda.is_available() else "cpu")
model_type = torch.float64

# ############### #
# Model Structure #
# ############### #

reg_A = 1e4
reg_B = 1e4
learning_rate = 1e-4

import modules.train_rnn as rnn
RNNFunc = rnn.RNN_tot_L2

# ######## #
# Training #
# ######## #

epochs = 1000
n_batches = 100
batch_duration = 200

test_freq = 100
visualize_train = False

# ########### #
# Inout Paths #
# ########### #

dataset_path = "dataset/parsed/tts-a-summer-train.csv"
params_path = "dataset/parsed/tts-a-params.csv"
out_directory = "out/train-rnn-L2"
