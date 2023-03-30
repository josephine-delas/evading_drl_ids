# Preprocessing function for AWID

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, MinMaxScaler

col_names = ['frame.interface_id',
 'frame.dlt',
 'frame.offset_shift',
 'frame.time_epoch',
 'frame.time_delta',
 'frame.time_delta_displayed',
 'frame.time_relative',
 'frame.len',
 'frame.cap_len',
 'frame.marked',
 'frame.ignored',
 'radiotap.version',
 'radiotap.pad',
 'radiotap.length',
 'radiotap.present.tsft',
 'radiotap.present.flags',
 'radiotap.present.rate',
 'radiotap.present.channel',
 'radiotap.present.fhss',
 'radiotap.present.dbm_antsignal',
 'radiotap.present.dbm_antnoise',
 'radiotap.present.lock_quality',
 'radiotap.present.tx_attenuation',
 'radiotap.present.db_tx_attenuation',
 'radiotap.present.dbm_tx_power',
 'radiotap.present.antenna',
 'radiotap.present.db_antsignal',
 'radiotap.present.db_antnoise',
 'radiotap.present.rxflags',
 'radiotap.present.xchannel',
 'radiotap.present.mcs',
 'radiotap.present.ampdu',
 'radiotap.present.vht',
 'radiotap.present.reserved',
 'radiotap.present.rtap_ns',
 'radiotap.present.vendor_ns',
 'radiotap.present.ext',
 'radiotap.mactime',
 'radiotap.flags.cfp',
 'radiotap.flags.preamble',
 'radiotap.flags.wep',
 'radiotap.flags.frag',
 'radiotap.flags.fcs',
 'radiotap.flags.datapad',
 'radiotap.flags.badfcs',
 'radiotap.flags.shortgi',
 'radiotap.datarate',
 'radiotap.channel.freq',
 'radiotap.channel.type.turbo',
 'radiotap.channel.type.cck',
 'radiotap.channel.type.ofdm',
 'radiotap.channel.type.2ghz',
 'radiotap.channel.type.5ghz',
 'radiotap.channel.type.passive',
 'radiotap.channel.type.dynamic',
 'radiotap.channel.type.gfsk',
 'radiotap.channel.type.gsm',
 'radiotap.channel.type.sturbo',
 'radiotap.channel.type.half',
 'radiotap.channel.type.quarter',
 'radiotap.dbm_antsignal',
 'radiotap.antenna',
 'radiotap.rxflags.badplcp',
 'wlan.fc.type_subtype',
 'wlan.fc.version',
 'wlan.fc.type',
 'wlan.fc.subtype',
 'wlan.fc.ds',
 'wlan.fc.frag',
 'wlan.fc.retry',
 'wlan.fc.pwrmgt',
 'wlan.fc.moredata',
 'wlan.fc.protected',
 'wlan.fc.order',
 'wlan.duration',
 'wlan.ra',
 'wlan.da',
 'wlan.ta',
 'wlan.sa',
 'wlan.bssid',
 'wlan.frag',
 'wlan.seq',
 'wlan.bar.type',
 'wlan.ba.control.ackpolicy',
 'wlan.ba.control.multitid',
 'wlan.ba.control.cbitmap',
 'wlan.bar.compressed.tidinfo',
 'wlan.ba.bm',
 'wlan.fcs_good',
 'wlan_mgt.fixed.capabilities.ess',
 'wlan_mgt.fixed.capabilities.ibss',
 'wlan_mgt.fixed.capabilities.cfpoll.ap',
 'wlan_mgt.fixed.capabilities.privacy',
 'wlan_mgt.fixed.capabilities.preamble',
 'wlan_mgt.fixed.capabilities.pbcc',
 'wlan_mgt.fixed.capabilities.agility',
 'wlan_mgt.fixed.capabilities.spec_man',
 'wlan_mgt.fixed.capabilities.short_slot_time',
 'wlan_mgt.fixed.capabilities.apsd',
 'wlan_mgt.fixed.capabilities.radio_measurement',
 'wlan_mgt.fixed.capabilities.dsss_ofdm',
 'wlan_mgt.fixed.capabilities.del_blk_ack',
 'wlan_mgt.fixed.capabilities.imm_blk_ack',
 'wlan_mgt.fixed.listen_ival',
 'wlan_mgt.fixed.current_ap',
 'wlan_mgt.fixed.status_code',
 'wlan_mgt.fixed.timestamp',
 'wlan_mgt.fixed.beacon',
 'wlan_mgt.fixed.aid',
 'wlan_mgt.fixed.reason_code',
 'wlan_mgt.fixed.auth.alg',
 'wlan_mgt.fixed.auth_seq',
 'wlan_mgt.fixed.category_code',
 'wlan_mgt.fixed.htact',
 'wlan_mgt.fixed.chanwidth',
 'wlan_mgt.fixed.fragment',
 'wlan_mgt.fixed.sequence',
 'wlan_mgt.tagged.all',
 'wlan_mgt.ssid',
 'wlan_mgt.ds.current_channel',
 'wlan_mgt.tim.dtim_count',
 'wlan_mgt.tim.dtim_period',
 'wlan_mgt.tim.bmapctl.multicast',
 'wlan_mgt.tim.bmapctl.offset',
 'wlan_mgt.country_info.environment',
 'wlan_mgt.rsn.version',
 'wlan_mgt.rsn.gcs.type',
 'wlan_mgt.rsn.pcs.count',
 'wlan_mgt.rsn.akms.count',
 'wlan_mgt.rsn.akms.type',
 'wlan_mgt.rsn.capabilities.preauth',
 'wlan_mgt.rsn.capabilities.no_pairwise',
 'wlan_mgt.rsn.capabilities.ptksa_replay_counter',
 'wlan_mgt.rsn.capabilities.gtksa_replay_counter',
 'wlan_mgt.rsn.capabilities.mfpr',
 'wlan_mgt.rsn.capabilities.mfpc',
 'wlan_mgt.rsn.capabilities.peerkey',
 'wlan_mgt.tcprep.trsmt_pow',
 'wlan_mgt.tcprep.link_mrg',
 'wlan.wep.iv',
 'wlan.wep.key',
 'wlan.wep.icv',
 'wlan.tkip.extiv',
 'wlan.ccmp.extiv',
 'wlan.qos.tid',
 'wlan.qos.priority',
 'wlan.qos.eosp',
 'wlan.qos.ack',
 'wlan.qos.amsdupresent',
 'wlan.qos.buf_state_indicated1',
 'wlan.qos.bit4',
 'wlan.qos.txop_dur_req',
 'wlan.qos.buf_state_indicated2',
 'data.len',
 'class']


def format_AWID(raw_train_path: str, raw_test_path: str, normalization: str = 'standard'):
    '''
    Preprocessing function for the AWID2 Dataset
    'https://icsdweb.aegean.gr/awid/awid2'

    Parameters
    ----------
    raw_train: String   
        Path for the CSV file: training set of the AWID Dataset
    raw_test: Stringdea
        Path for the CSV file: testing set of the AWID Dataset
    normalization: String
        Either 'maxmin' or 'standard' normalization
    
    Outputs
    -------
    formated_train: pd.DataFrame   
        Formated training set of the AWID Dataset
    formated_test: pd.Dataframe
        Formated testing set of the AWID Dataset
    means: pd.Dataframe
        means of the initial training dataframe's numerical columns
    stds: pd.Dataframe
        stds of the initial training dataframe's numerical columns
    '''
    df_train = pd.read_csv(raw_train_path, header=None, names=col_names) 
    df_test = pd.read_csv(raw_test_path, header=None, names=col_names)
    
    # Concatenate
    df = pd.concat([df_train, df_test])
    train_indx = df_train.shape[0] # To separate training and testing data

    # Replace the '?' values with None
    df.replace({"?": np.nan}, inplace=True)

    # Remove columns with more than 50% null data
    col_to_remove = df.iloc[0:train_indx].columns[df.iloc[0:train_indx].isna().mean() >= 0.5]
    print("Removed " + str(col_to_remove.shape[0]) + " columns with > 50% null data")
    df.drop(col_to_remove, axis=1, inplace=True)

    # Remove columns with no variation 
    col_to_remove = df.iloc[0:train_indx].columns[df.iloc[0:train_indx].nunique()==1]
    print("Removed " + str(col_to_remove.shape[0]) + " columns with no variation")
    df.drop(col_to_remove, axis=1, inplace=True)
    
    # Print df shape
    print("DataFrame's current shape: " + str(df.shape))

    # Remove columns with almost only unique values
    col_to_remove = df.iloc[0:train_indx].columns[df.iloc[0:train_indx].nunique()>train_indx*0.99]
    print("Removed " + str(col_to_remove.shape[0]) + " columns with only unique values (time)")
    df.drop(col_to_remove, axis=1, inplace=True)

    # remove columns with IP adresses 
    col_to_remove = ['wlan.ra', 'wlan.da', 'wlan.ta', 'wlan.sa', 'wlan.bssid']
    print("Removed " + str(len(col_to_remove)) + " columns containing IP adresses")
    df.drop(col_to_remove, axis=1, inplace=True)

    # convert str into int values
    for col in df:
        if col != 'class':
            df[col] = df[col].apply(lambda x : int(x,0) if type(x)==str else x)

    # drop the remaining columns with only one value
    col_to_remove = []
    for col in df:
        if len(df.iloc[0:train_indx][col].dropna().unique())==1:
            col_to_remove.append(col)
    df.drop(col_to_remove, axis=1, inplace=True)
    print("Removed another " + str(len(col_to_remove)) + " columns with no variation and null data")

    formated_train = df.iloc[0:train_indx]
    formated_test = df.iloc[train_indx:df.shape[0]]

    # Remove the remaining rows with missing values (doesn't change the class distribution)
    formated_train.dropna(inplace=True)
    formated_test.dropna(inplace=True)

    train_indx = formated_train.shape[0] # new separator index
    df = pd.concat([formated_train, formated_test]) # concatenated dataframe (train+test)
    df.drop(['radiotap.length'], axis=1, inplace=True) # This column is no longer relevant (constant value)

    print('Training test shape : ', formated_train.shape)
    print('Testing test shape : ', formated_test.shape)

    boolean_cols = ['radiotap.present.tsft', 'radiotap.present.flags', 'radiotap.present.channel', 'radiotap.present.dbm_antsignal', 'radiotap.present.antenna', 'radiotap.present.rxflags', 'radiotap.channel.type.cck', 'radiotap.channel.type.ofdm', 'wlan.fc.frag', 'wlan.fc.retry', 'wlan.fc.pwrmgt', 'wlan.fc.moredata', 'wlan.fc.protected']
    categorical_cols = ['radiotap.datarate', 'radiotap.channel.freq', 'wlan.fc.type', 'wlan.fc.subtype', 'wlan.fc.ds', 'wlan.frag']
    numerical_cols = ['frame.time_delta', 'frame.time_delta_displayed', 'frame.len', 'frame.cap_len', 'radiotap.dbm_antsignal', 'wlan.fc.type_subtype', 'wlan.duration', 'wlan.seq']
    
    # Normalize numerical columns
    if normalization == 'maxmin':
        scaler = MinMaxScaler()
    elif normalization == 'standard':
        scaler = StandardScaler()
    else:
        raise ValueError("normalization parameter should be 'maxmin' or 'standard', but another value was found")
    means = df[numerical_cols][0:train_indx].mean() # Store means and std before normalization
    stds = df[numerical_cols][0:train_indx].std()
    scaler.fit(df[numerical_cols][0:train_indx]) # Fit only on the training data to avoid bias
    scaled_array = scaler.transform(df[numerical_cols])
    df[numerical_cols] = scaled_array

    # Convert boolean columns to bool type
    df[boolean_cols] = df[boolean_cols].astype('bool')

    # One-hot encoding for categorical columns
    for col in categorical_cols:
        df = pd.concat([pd.get_dummies(df[col], prefix = col).astype('bool'), df], axis=1)
        df.drop([col], axis=1, inplace=True)

    df.rename(columns={"class": "labels"}, inplace=True)

    print(df.describe())

    # Split again between training and testing data
    formated_train = df.iloc[0:train_indx]
    formated_test = df.iloc[train_indx:df.shape[0]]

    return formated_train, formated_test, means, stds