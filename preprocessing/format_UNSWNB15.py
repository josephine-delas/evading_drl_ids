import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def format_UNSWNB15(raw_train_path: str, raw_test_path: str, normalization: str = 'standard'):

    # import the dataset into pandas DataFrames
    df_training = pd.read_csv(raw_train_path)
    df_testing = pd.read_csv(raw_test_path)

    # stack the training and testing sets
    df_data = pd.concat([df_training, df_testing], axis=0)

    # remove the columns 'id' and 'attack_cat'
    df_data.drop('id', inplace=True, axis=1)
    df_data.drop('label', inplace=True, axis=1)

    # 'is_ftp_login' should be a binary feature, we remove the instances that hold the values 2 and 4
    df_data = df_data[df_data['is_ftp_login'] != 2]
    df_data = df_data[df_data['is_ftp_login'] != 4]

    categorical_features = ['state', 'service', 'proto']
    df_data = pd.get_dummies(df_data, columns=categorical_features, prefix=categorical_features, prefix_sep=":")
    # move the labels back to the last column in lowercase
    df_data['labels'] = df_data.pop('attack_cat').apply(lambda x: x.lower())

    # Min-Max normalization on the non-binary features
    # the min and max values are computed on the training set
    continuous_features = ['dur', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'sttl', 'dttl', 'sload', 'dload', 'sloss', 'dloss', 'sinpkt', 'dinpkt', 'sjit', 'djit', 'swin', 'stcpb', 'dtcpb', 'dwin', 'tcprtt', 'synack', 'ackdat', 'smean', 'dmean', 'trans_depth', 'response_body_len', 'ct_srv_src', 'ct_state_ttl', 'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'ct_ftp_cmd', 'ct_flw_http_mthd', 'ct_src_ltm', 'ct_srv_dst']
    min = df_data[:df_training.shape[0]][continuous_features].min()
    max = df_data[:df_training.shape[0]][continuous_features].max()
    df_data[continuous_features] = (df_data[continuous_features] - min) / (max - min)

    # Normalize numerical columns
    if normalization == 'maxmin':
        scaler = MinMaxScaler()
    elif normalization == 'standard':
        scaler = StandardScaler()
    else:
        raise ValueError("normalization parameter should be 'maxmin' or 'standard', but another value was found")
    means = df_data[:df_training.shape[0]][continuous_features].mean() # Store means and std before normalization
    stds = df_data[:df_training.shape[0]][continuous_features].std()
    scaler.fit(df_data[:df_training.shape[0]][continuous_features]) # Fit only on the training data to avoid bias
    scaled_array = scaler.transform(df_data[continuous_features])
    df_data[continuous_features] = scaled_array

    # split training and testing sets
    formated_train = df_data[:df_training.shape[0]]    
    formated_test = df_data[df_training.shape[0]:]

    return formated_train, formated_test, means, stds