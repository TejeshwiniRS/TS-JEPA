import numpy as np 
import pandas as pd 
import wfdb
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import configparser

# Load Configuration Data from config.ini
config = configparser.ConfigParser()
config.read('../config.ini')
data_file_path = config['DATA']['dataset_path']
sampling_rate = int(config['DATA']['sampling_rate'])

print(f"Loading data from {data_file_path} with sampling rate {sampling_rate} Hz")
df = pd.read_csv(f"{data_file_path}/ptbxl_database.csv", index_col="ecg_id")
df.scp_codes = df.scp_codes.apply(lambda x: eval(x) if isinstance(x, str) else {})


def load_raw_signals(df, sampling_rate, path):
    if sampling_rate != 100:
        raise ValueError("Only 100 Hz sampling rate is supported.")
    data = []
    for file in df.filename_lr:
        data.append(wfdb.rdsamp(f"{path}/{file}"))
    return np.array([signal for signal, _ in data])

X_raw = load_raw_signals(df, sampling_rate, data_file_path) # shape: (n_samples, n_timesteps, n_channels)

X = X_raw.transpose(0, 2, 1) # shape: (n_samples, n_channels, n_timesteps)
X = X.astype(np.float32)

# Normalization 
mean = X.mean(axis=(0, 2), keepdims=True)
std = X.std(axis=(0, 2), keepdims=True)
X = (X - mean) / std

scp_df = pd.read_csv(f"{data_file_path}/scp_statements.csv", index_col=0)
diag_map = scp_df[scp_df.diagnostic == 1.0].index ## Filter only rows where ECG is diagnostic and not form or rhythm 

def get_super_class(codes):
    superclass_labels = []
    for c in codes:
        if (c in scp_df.index and scp_df.loc[c, 'diagnostic'] == 1.0):
            superclass_labels.append(scp_df.loc[c, 'diagnostic_class'])
    return superclass_labels
df["superclass"] = df.scp_codes.apply(get_super_class)
mlb = MultiLabelBinarizer(sparse_output=False)
y = np.array(mlb.fit_transform(df.superclass), dtype=np.float32)

train_idx = df[df.strat_fold <= 8].index - 1
val_idx = df[df.strat_fold == 9].index - 1
test_idx = df[df.strat_fold == 10].index - 1

X_train, y_train = X[train_idx], y[train_idx]
X_val,   y_val   = X[val_idx],   y[val_idx]
X_test,  y_test  = X[test_idx],  y[test_idx]

# ── 7. Save to .npy files ─────────────────────────────────────────────────────
np.save("transformed_data/X_ecg_train.npy", X_train)
np.save("transformed_data/X_ecg_val.npy",   X_val)
np.save("transformed_data/X_ecg_test.npy",  X_test)
np.save("transformed_data/y_ecg_train.npy", y_train)
np.save("transformed_data/y_ecg_val.npy",   y_val)
np.save("transformed_data/y_ecg_test.npy",  y_test)

# Save normalization stats for inference-time use
np.save("transformed_data/norm_ecg_mean.npy", mean)
np.save("transformed_data/norm_ecg_std.npy",  std)
