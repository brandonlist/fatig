import scipy.io as sio
import os
from sklearn.model_selection import KFold, LeavePOut
import numpy as np
from vfe.io.eeglab import read_epochs_from_eeglab_singlefile
from vfe.util.EEGDataSet import EEGDATA
from torch.utils.data import DataLoader
import copy
from vfe.util.EEGDataSet import EEGDataSet
from Nets.MoletNet.MorletFatigueNet import MorletInceptionNet,MorletInceptionNetSA
import torch.nn.functional as F
import torch
import tqdm
from vfe.util.trainSchedule import Scheduler
from vfe.util.summaryUtils import Summary
from vfe.util.metricUtils import Metric
from train_frame_pac.train import hc_train

def read_data_from_filelist(dataroot, file_name_list):
    # load data
    s_data = sio.loadmat('./data/subjective_score/subject.mat')
    subject_data = s_data['subjective']  # note that row 4 data is unaviale
    data_root = dataroot
    # build a 0 * 30 (channel) * 375 (time samples)
    X = np.ndarray((0, 30, 375))
    # fatigue label
    Z = np.ndarray((0))
    for filename in file_name_list:
        file_full_pathname = os.path.join(data_root, filename)
        epoch = read_epochs_from_eeglab_singlefile(file_full_pathname, 'vep_phase')
        # epoch.filter(4,13)
        X = np.concatenate((X, (epoch.get_data() * 1e6).astype(np.float32)))
        # generate fatigue label
        subject_index = int(filename[:-38])
        viewing_time = epoch.events[:, 2]
        z = []
        for i in range(len(viewing_time)):
            temp = viewing_time[i]
            if temp > 10:
                temp = temp - 10
            z += [subject_data[subject_index - 1, temp - 1]]
        Z = np.concatenate((Z, z))

    # load data end
    Z[np.where(Z == 1)] = 0
    Z[np.where(Z == 2)] = 0
    Z[np.where(Z == 3)] = 1
    Z[np.where(Z == 4)] = 2
    Z[np.where(Z == 5)] = 2

    return X, Z


train_name = 'Deep4NetSA_single_sub'
log_fold = '20211231'


from braindecode.models.deep4 import Deep4Net,Deep4NetSA
from braindecode.models.eegnet import EEGNetv4,EEGNetv4SA
from braindecode.models.shallow_fbcsp import ShallowFBCSPNet,ShallowFBCSPNetSA
# model1 = ShallowFBCSPNet(in_chans=30,n_classes=3,input_time_length=375,final_conv_length='auto').create_network()
# model1 = ShallowFBCSPNetSA(in_chans=30,n_classes=3,input_time_length=375,final_conv_length='auto',drop_prob=0.7)
# model1 = Deep4Net(in_chans=30,n_classes=3,input_time_length=375,final_conv_length='auto').create_network()
model1 = Deep4NetSA(in_chans=30,n_classes=3,input_time_length=375,final_conv_length='auto')
# model1 = EEGNetv4(in_chans=30,n_classes=3,input_time_length=375,final_conv_length='auto').create_network()
# model1 = EEGNetv4SA(in_chans=30,n_classes=3,input_time_length=375,final_conv_length='auto',drop_prob=0.5)
# model1 = MorletInceptionNetSA()
# model1 = MorletInceptionNet(drop0=0.1,drop1=0.1)
# load data
s_data = sio.loadmat('./data/subjective_score/subject.mat')
subject_data = s_data['subjective']  # note that row 4 data is unaviale
data_root = './data/erp/'
# read .set file
file_list = os.listdir(data_root)
kp = LeavePOut(1)


test_accs = np.zeros((20,5))

for j in np.arange(len(file_list)):
    sub_i = int(file_list[j].split('subject')[0]) - 1

    if sub_i==5:
        continue

    x, z = read_data_from_filelist(dataroot=data_root, file_name_list=np.array(file_list)[np.array([j])])
    eeg_data = EEGDATA(X=x, Z=z, percentage=(0.8, 0.2, 0))
    x_data = eeg_data.get_trainset().get_x().numpy()
    y_data = eeg_data.get_trainset().get_y().numpy()


    testloader = DataLoader(eeg_data.get_testset(), batch_size=64, drop_last=True)
    kf = KFold(n_splits=5)
    i = 1
    for train_index, val_index in kf.split(x_data):
        print('processing fold {}'.format(i))
        x_train_data, x_val_data = x_data[train_index], x_data[val_index]
        y_train_data, y_val_data = y_data[train_index], y_data[val_index]
        traindataset = EEGDataSet(x_train_data, y_train_data)
        valdataset = EEGDataSet(x_val_data, y_val_data)
        train_loader = DataLoader(traindataset, batch_size=64, sampler=traindataset.get_sampler(), drop_last=True)
        val_loader = DataLoader(valdataset, batch_size=64, drop_last=True)
        train_name1 = train_name + "{}_KF{}".format(sub_i, i)
        model = copy.deepcopy(model1)
        test_accs[sub_i][i-1] =  hc_train(logfold=log_fold, train_name=train_name1, train_loader=train_loader, val_loader=val_loader,
                     testloader=testloader, model1=model, l1_lambda=0.1)

        i += 1

