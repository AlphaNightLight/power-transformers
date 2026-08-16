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

reg_A = 0.0
reg_B = 0.0
learning_rate = 1e-3

import modules.train_rnn as rnn
RNNFunc = rnn.RNN_b

# ######## #
# Training #
# ######## #

epochs = 500
n_batches = 50
batch_duration = 100

test_freq = 20
visualize_train = False

# ########### #
# Inout Paths #
# ########### #

dataset_path = "dataset/parsed/tts-a-summer-train.csv"
params_path = "dataset/parsed/tts-a-params.csv"
out_directory = "out/train-rnn-b"
