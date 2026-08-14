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

reg_A = 5.0
reg_B = 5.0
learning_rate = 1e-4

import modules.train_rnn as rnn
RNNFunc = rnn.RNN_abcdef #@# rnn.RNN_tot_L05

# ######## #
# Training #
# ######## #

epochs = 50 #@# 1000 # 500
n_batches = 10 #@# 100 # 50
batch_duration = 10 #@# 200 # 100

test_freq = 10 #@# 100 # 20
visualize_train = True

# ########### #
# Inout Paths #
# ########### #

dataset_path = "dataset/tts-a-summer.csv" #@# "dataset/tts-c-summer.csv"
params_path = "dataset/tts-a-params.csv" #@# "dataset/tts-c-params.csv"
out_directory = "trainAAA" #@# "train-L05"
