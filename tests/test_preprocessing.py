import pandas as pd
import argparse

from pathlib import Path
import sys
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from preprocessing import FormatKDD, FormatAWID

if __name__ == '__main__':

    print('Start...')

    ####----Parameters----####
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", required=True, default="KDD", help="Dataset to use: 'KDD' or 'AWID'.")
    parser.add_argument("-s", "--save", default=0, type=int, help="Save newly preprocessed data (1) or use pre-existing one (0)")
    parser.add_argument("-v", "--verbose", default=1, type=int, help="Print information on the preprocessed dataset (1) or not (0)")
    args = parser.parse_args()
    dataset= args.data
    save = args.save
    verbose = args.verbose

    # Get dataset paths
    raw_train_path_KDD = '../datasets/KDD/KDDTrain+.txt'
    raw_test_path_KDD = '../datasets/KDD/KDDTest+.txt'
    raw_test_path_AWID = '../datasets/AWID/AWID-CLS-R-Tst.csv'
    raw_train_path_AWID = '../datasets/AWID/AWID-CLS-R-Trn.csv'

    if save : 
        if dataset=='KDD':
            formated_train_KDD, formated_test_KDD, means, stds = FormatKDD(raw_train_path_KDD, raw_test_path_KDD, normalization='standard')
            formated_train_KDD.to_parquet('../datasets/KDD/formated_KDD_train.parquet',index=False)
            formated_test_KDD.to_parquet('../datasets/KDD/formated_KDD_test.parquet',index=False)
            means.to_csv('../datasets/KDD/means_KDD.csv', sep=',', index=False)
            stds.to_csv('../datasets/KDD/stds_KDD.csv', sep=',', index=False)
            
        elif dataset=='AWID':
            formated_train_AWID, formated_test_AWID, means, stds = FormatAWID(raw_train_path=raw_train_path_AWID, raw_test_path=raw_test_path_AWID, normalization = 'standard')
            formated_train_AWID.to_parquet('../datasets/AWID/formated_AWID_train.parquet',index=False)
            formated_test_AWID.to_parquet('../datasets/AWID/formated_AWID_test.parquet',index=False)
            means.to_csv('../datasets/AWID/means_AWID.csv', sep=',', index=True)
            stds.to_csv('../datasets/AWID/stds_AWID.csv', sep=',', index=True)

        else:
            raise ValueError("Unknown dataset. Received {}, should be 'KDD' or 'AWID'".format(dataset))
    
    print("Loading...")
    if dataset=='KDD':
        train_df = pd.read_parquet('../datasets/KDD/formated_KDD_train.parquet')
        test_df = pd.read_parquet('../datasets/KDD/formated_KDD_test.parquet')

    elif dataset=='AWID':
        train_df = pd.read_parquet('../datasets/AWID/formated_AWID_train.parquet')
        test_df = pd.read_parquet('../datasets/AWID/formated_AWID_test.parquet')
    
    if verbose:

        print("Training set preview: ")
        print(train_df.head(4))
        print(train_df.shape)

        print("Testing set preview: ")
        print(test_df.head(4))
        print(test_df.shape)
        
        print("Maximums: ")
        print(train_df.max())
        print(test_df.max())

        print("Minimums: ")
        print(train_df.min())
        print(test_df.min())

        print("Means: ")
        print(train_df.mean())
        print(test_df.mean())

        print("Standard deviation: ")
        print(train_df.std())
        print(test_df.std())
    
    print('...End')