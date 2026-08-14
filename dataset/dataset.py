# ####### #
# Imports #
# ####### #

import pandas as pd

from modules.inout import mkdir

dataset_dir = "parsed"
mkdir(dataset_dir)





# #### #
# TTSa #
# #### #

# Summer

df = pd.read_excel(
    io="CIGRE/TTSa/TTSa16-timeseries-training-VTP-summerONAF2.xlsx",
    sheet_name="data"
)
df = df[[
    "time",
    "load",
    "t_amb",
    "t_tl",
    "t_hs"
]]
df.columns = ["t_d", "K", "theta_a", "theta_o", "theta_h"]
df = df.round({"K": 4, "theta_a": 1, "theta_o": 1, "theta_h": 1})

df.to_csv(
    path_or_buf=dataset_dir+"/tts-a-summer.csv",
    sep=";",
    index=False,
    date_format="%d/%m/%Y %H:%M"
)

df_train = df.iloc[0:-5001]
df_train.to_csv(
    path_or_buf=dataset_dir+"/tts-a-summer-train.csv",
    sep=";",
    index=False,
    date_format="%d/%m/%Y %H:%M"
)

df_test = df.iloc[-5001:-1]
df_test.to_csv(
    path_or_buf=dataset_dir+"/tts-a-summer-test.csv",
    sep=";",
    index=False,
    date_format="%d/%m/%Y %H:%M"
)
