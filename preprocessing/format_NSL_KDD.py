# Preprocessing function for NSL-KDD

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

col_names = ["duration","protocol_type","service","flag","src_bytes",
            "dst_bytes","land_f","wrong_fragment","urgent","hot","num_failed_logins",
            "logged_in","num_compromised","root_shell","su_attempted","num_root",
            "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
            "is_host_login","is_guest_login","count","srv_count","serror_rate",
            "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
            "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
            "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
            "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
            "dst_host_rerror_rate","dst_host_srv_rerror_rate","labels","dificulty"]

attack_map =   { 'normal': 'normal',
                        
                        'back': 'DoS',
                        'land': 'DoS',
                        'neptune': 'DoS',
                        'pod': 'DoS',
                        'smurf': 'DoS',
                        'teardrop': 'DoS',
                        'mailbomb': 'DoS',
                        'apache2': 'DoS',
                        'processtable': 'DoS',
                        'udpstorm': 'DoS',
                        
                        'ipsweep': 'Probe',
                        'nmap': 'Probe',
                        'portsweep': 'Probe',
                        'satan': 'Probe',
                        'mscan': 'Probe',
                        'saint': 'Probe',
                    
                        'ftp_write': 'R2L',
                        'guess_passwd': 'R2L',
                        'imap': 'R2L',
                        'multihop': 'R2L',
                        'phf': 'R2L',
                        'spy': 'R2L',
                        'warezclient': 'R2L',
                        'warezmaster': 'R2L',
                        'sendmail': 'R2L',
                        'named': 'R2L',
                        'snmpgetattack': 'R2L',
                        'snmpguess': 'R2L',
                        'xlock': 'R2L',
                        'xsnoop': 'R2L',
                        'worm': 'R2L',
                        
                        'buffer_overflow': 'U2R',
                        'loadmodule': 'U2R',
                        'perl': 'U2R',
                        'rootkit': 'U2R',
                        'httptunnel': 'U2R',
                        'ps': 'U2R',    
                        'sqlattack': 'U2R',
                        'xterm': 'U2R'
                    }

def format_NSL_KDD(raw_train_path: str, raw_test_path: str, normalization: str='maxmin'):
    '''
    Preprocessing function for the NSL-KDD Dataset
    'https://www.unb.ca/cic/datasets/nsl.html'

    Parameters
    ----------
    raw_train: String   
        Path for the CSV file: training set of the NSL-KDD dataSet
    raw_test: String
        Path for the CSV file: testing set of the NSL-KDD Dataset
    normalization: String
        Either 'maxmin' or 'standard' normalization
    
    Outputs
    -------
    formated_train: pd.DataFrame   
        Formated training set of the NSL-KDD Dataset
    formated_test: pd.Dataframe
        Formated testing set of the NSL-KDD Dataset
    means: pd.Dataframe
        means of the initial training dataframe's numerical columns
    stds: pd.Dataframe
        stds of the initial training dataframe's numerical columns
    '''
    # Load the data from the CSV files
    df_train = pd.read_csv(raw_train_path,sep=',',names=col_names,index_col=False)
    df_test = pd.read_csv(raw_test_path,sep=',',names=col_names,index_col=False)

    # Concatenate
    df = pd.concat([df_train, df_test])
    train_indx = df_train.shape[0] # To separate training and testing data

    # Remove the 'dificulty' and 'num_outbound_cmds' colummns
    df.drop('dificulty', axis=1, inplace=True)
    df.drop('num_outbound_cmds', axis=1, inplace=True)

    # 0-1 values for the 'su_attempted' column
    df['su_attempted'] = df['su_attempted'].replace(2.0, 0.0)

    # Separate categorical and numerical data
    categorical = df[['protocol_type', 'service', 'flag', 'labels', 'su_attempted', 'is_guest_login', 'is_host_login']]
    df = df.drop(['protocol_type', 'service', 'flag', 'labels', 'su_attempted', 'is_guest_login', 'is_host_login'], axis=1)
    col_num = df.columns
    idx_num = df.index

    # Normalize numerical columns
    if normalization == 'maxmin':
        scaler = MinMaxScaler()
    elif normalization == 'standard':
        scaler = StandardScaler()
    else:
        raise ValueError("normalization parameter should be 'maxmin' or 'standard', but another value was found")
    scaler.fit(df[0:train_indx]) # Fit only on the training data to avoid bias
    means = df[0:train_indx].mean() # Store means and std before normalization
    stds = df[0:train_indx].std()
    scaled_array = scaler.transform(df)
    

    df = pd.DataFrame(scaled_array, columns = col_num, index = idx_num)

    # Add one-hot encoding for the categorical data
    df = pd.concat([df, categorical[['su_attempted', 'is_guest_login', 'is_host_login']]], axis=1)
    df = pd.concat([df, pd.get_dummies(categorical['protocol_type'])], axis=1)
    df = pd.concat([df, pd.get_dummies(categorical['service'])], axis=1)
    df = pd.concat([df, pd.get_dummies(categorical['flag'])], axis=1)

    # Add the label column, unmodified
    df = pd.concat([df, categorical['labels']], axis=1)

    # Separate training and testing sets
    formated_train = df.iloc[0:train_indx]
    formated_test = df.iloc[train_indx:df.shape[0]]

    return formated_train, formated_test, means, stds