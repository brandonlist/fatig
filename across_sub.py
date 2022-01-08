import torch.nn.functional as F
import torch
import tqdm
from Nets.MoletNet.MorletFatigueNet import MorletInceptionNet,MorletInceptionNetSA
from vfe.util.EEGDataSet import EEGDATA
from vfe.util.EEGDataSet import EEGDataSet
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold, LeavePOut
import copy
from vfe.util.trainSchedule import Scheduler
from vfe.util.summaryUtils import Summary
from vfe.util.metricUtils import Metric

from train_frame_pac.train import hc_train


from Nets.SelfAttention.selfAttentionCNN import SelfAttentionShallowConvNet
from DNN_printer import DNN_printer
model1 = MorletInceptionNetSA()
model1 = MorletInceptionNet(drop0=0.3,drop1=0.3)

from braindecode.models.deep4 import Deep4Net
from braindecode.models.eegnet import EEGNetv4
from braindecode.models.shallow_fbcsp import ShallowFBCSPNet
# model1 = ShallowFBCSPNet(in_chans=30,n_classes=3,input_time_length=375,final_conv_length='auto').create_network()
# model1 = Deep4Net(in_chans=30,n_classes=3,input_time_length=375,final_conv_length='auto').create_network()
# model1 = EEGNetv4(in_chans=30,n_classes=3,input_time_length=375,final_conv_length='auto').create_network()


k = 5
batch_size = 64
train_name = 'MorletInception_across_sub'
logfold = '20211228'
eeg_data = EEGDATA(data_path='./data/npy/save_X.npy', label_path='./data/npy/save_Z.npy', percentage=(0.8, 0.2, 0))
x_data = eeg_data.get_trainset().get_x().numpy()
y_data = eeg_data.get_trainset().get_y().numpy()
testloader = DataLoader(eeg_data.get_testset(), batch_size=batch_size, drop_last=True)
kf = KFold(n_splits=k)
i = 1

test_accs = []
selected_fold = [1]  #to test param

for train_index, val_index in kf.split(x_data):
    if i in selected_fold:
        print('processing fold {}'.format(i))
        x_train_data, x_val_data = x_data[train_index], x_data[val_index]
        y_train_data, y_val_data = y_data[train_index], y_data[val_index]
        traindataset = EEGDataSet(x_train_data, y_train_data)
        valdataset = EEGDataSet(x_val_data, y_val_data)
        train_loader = DataLoader(traindataset, batch_size=batch_size, sampler=traindataset.get_sampler(), drop_last=True)
        val_loader = DataLoader(valdataset, batch_size=batch_size, drop_last=True)
        train_name1 = train_name + "{}".format(i)
        model = copy.deepcopy(model1)
        test_accs.append(hc_train(logfold=logfold,train_name=train_name1,train_loader=train_loader,val_loader=val_loader,testloader=testloader,model1=model,l1_lambda=0.1))
    i += 1

