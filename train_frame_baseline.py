import torch.nn.functional as F
import torch
import tqdm
from vfe.util.EEGDataSet import EEGDATA
from torch.utils.data import DataLoader
from vfe.util.summaryUtils import Summary
from vfe.util.metricUtils import Metric
from Nets.MoletNet.MorletFatigueNet import MorletInceptionNet
from sklearn.model_selection import KFold, LeavePOut
from vfe.util.EEGDataSet import EEGDataSet
import os
import scipy.io as sio
from vfe.io.eeglab import read_epochs_from_eeglab_singlefile
import numpy as np
import copy
from collections import OrderedDict
from braindecode.datautil.splitters import split_into_two_sets, concatenate_two_sets
from braindecode.datautil.signal_target import SignalAndTarget
from braindecode.torch_ext.schedulers import ScheduledOptimizer, CosineAnnealing
from braindecode.models.shallow_fbcsp import ShallowFBCSPNet
from braindecode.datasets.bcic_iv_2a import BCICompetition4Set2A
from braindecode.models.deep4 import Deep4Net
from braindecode.models.eegnet import EEGNetv4
from braindecode.mne_ext.signalproc import mne_apply
from braindecode.datautil.signalproc import (bandpass_cnt,
                                             exponential_running_standardize)
from braindecode.datautil.trial_segment import create_signal_target_from_raw_mne

import warnings
warnings.filterwarnings("ignore")

def yk_kfold_train_MI_across_subject(trainname, logfold, model1, k=5):
    train_name = trainname
    log_fold = logfold
    # load data
    subject_id_list = list(range(9))
    all_train_set, all_test_set = load_multi_subject_MI(data_folder='./data/bci2a/', subject_id_list=subject_id_list)

    eeg_data = EEGDATA(X=all_train_set.X, Z=all_train_set.y, percentage=(1.0, 0, 0))
    eeg_test_data = EEGDATA(X=all_test_set.X, Z=all_test_set.y, percentage=(1.0, 0, 0))
    x_data = eeg_data.get_trainset().get_x().numpy()
    y_data = eeg_data.get_trainset().get_y().numpy()
    x_test_data = eeg_test_data.get_trainset().get_x().numpy()
    y_test_data = eeg_test_data.get_trainset().get_y().numpy()
    kf = KFold(n_splits=k)
    i = 1
    for train_index, val_index in kf.split(x_data):
        print('processing fold {}'.format(i))
        x_train_data, x_val_data = x_data[train_index], x_data[val_index]
        y_train_data, y_val_data = y_data[train_index], y_data[val_index]
        traindataset = EEGDataSet(x_train_data, y_train_data)
        valdataset = EEGDataSet(x_val_data, y_val_data)
        testset = EEGDataSet(x_test_data, y_test_data)
        train_loader = DataLoader(traindataset, batch_size=64, sampler=traindataset.get_sampler(), drop_last=True)
        val_loader = DataLoader(valdataset, batch_size=64, drop_last=True)
        test_loader = DataLoader(testset, batch_size=64, drop_last=False)
        train_name1 = train_name + '{}'.format(i)
        model = copy.deepcopy(model1)
        yk_train(logfoldname=log_fold, trainname=train_name1, modell=model, trainloader=train_loader, valloader=val_loader, testloader=test_loader)
        i += 1

def yk_kfold_train_MI_specific_subject(trainname, logfold, model1, k=5):
    train_name = trainname
    log_fold = logfold
    # load data
    for iid in range(9):
        subject_id = iid+1
        data_set, test_set = load_single_subject_MI(data_folder='./data/bci2a/', subject_id=subject_id)

        eeg_data = EEGDATA(X=data_set.X, Z=data_set.y, percentage=(1.0, 0, 0))
        eeg_test_data = EEGDATA(X=test_set.X, Z=test_set.y, percentage=(1.0, 0, 0))
        x_data = eeg_data.get_trainset().get_x().numpy()
        y_data = eeg_data.get_trainset().get_y().numpy()
        x_test_data = eeg_test_data.get_trainset().get_x().numpy()
        y_test_data = eeg_test_data.get_trainset().get_y().numpy()

        kf = KFold(n_splits=k)
        i = 1
        for train_index, val_index in kf.split(x_data):
            print('processing fold {}'.format(i))
            x_train_data, x_val_data = x_data[train_index], x_data[val_index]
            y_train_data, y_val_data = y_data[train_index], y_data[val_index]
            traindataset = EEGDataSet(x_train_data, y_train_data)
            valdataset = EEGDataSet(x_val_data, y_val_data)
            testset = EEGDataSet(x_test_data, y_test_data)
            train_loader = DataLoader(traindataset, batch_size=64, sampler=traindataset.get_sampler(), drop_last=False)
            val_loader = DataLoader(valdataset, batch_size=64, drop_last=False)
            test_loader = DataLoader(testset, batch_size=64, drop_last=False)
            train_name1 = train_name + '_sub{}'.format(subject_id) + '_KF{}'.format(i)
            model = copy.deepcopy(model1)
            yk_train(logfoldname=log_fold, trainname=train_name1, modell=model, trainloader=train_loader, valloader=val_loader, testloader=test_loader)
            i += 1

def yk_train(logfoldname, trainname, modell, trainloader, valloader, testloader,lr0=2*(2e-3)):
    # Parameters
    train_name=trainname
    filename1 = './' + trainname + '.t7'
    # init mode
    model = modell.cuda()
    optimizer1 = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0)
    n_epochs = 200
    n_updates_per_epoch = len(list(trainloader))
    scheduler = CosineAnnealing(n_epochs * n_updates_per_epoch)
    optimizer = ScheduledOptimizer(scheduler, optimizer1, schedule_weight_decay=True)
    yk_summary = Summary(train_name, logfoldname)
    yk_metric = Metric()
    # best metrics
    best_state = {
        'state':model.state_dict(),
        'epoch':0
    }
    model_for_save = list()
    for epoch in range(n_epochs):
        running_loss = 0.0
        model.train()
        for i, (images, labels) in tqdm.tqdm(enumerate(trainloader)):
            # braindecode
            images = images.permute((0,2,3,1))

            images = images.float().cuda()
            labels = labels.squeeze().cuda()
            optimizer.zero_grad()
            output = model(images)
            loss = F.nll_loss(output, labels)
            loss.backward()
            running_loss += loss.detach()
            optimizer.step()
        print('epoch[%d/%d] train loss:%.5f'%(epoch, 200, running_loss/(i+1)))
        yk_summary.addTrainLoss(running_loss/(i+1), epoch)
    #     ##################################eval epoch#############################################
        if epoch % 1==0:
            model.eval()
            temp_auc = 0.0
            temp_kappa = 0.0
            eval_loss=0.0
            for i, (images, labels) in enumerate(valloader):
                #  兼容braindecode
                images = images.permute((0, 2, 3, 1))

                images = images.float().cuda()
                labels = labels.cuda()
                output = model(images)
                loss = F.nll_loss(output, labels)
                eval_loss += loss.detach()
                temp_auc += yk_metric.vfe_accuracy_score(labels.cpu(), output.detach().cpu())
                temp_kappa += yk_metric.vfe_kappa(labels.cpu(), output.detach().cpu())
                # print(labels.size())
                # schedule.store_val_result(epoch,i,unNorm(images.cpu().data[0]),labels.cpu().data[0],output.cpu().data[0])
            print('epoch[%d/%d]  val loss:%.5f' % (epoch, 200, eval_loss / (i+1)))
            print('              val acc:%.2f' % (100 * temp_auc / (i+1)))
            tt = eval_loss/(i+1)
            yk_summary.addValLoss(tt, epoch)
            yk_summary.addValAUC(temp_auc / (i+1), epoch)
            yk_summary.addValKappa(temp_kappa/(i+1), epoch)

            if testloader is None:
                continue
            test_auc = 0.0
            test_loss = 0.0
            for i, (images, labels) in enumerate(testloader):
                #  兼容braindecode
                images = images.permute((0, 2, 3, 1))
                images = images.float().cuda()
                labels = labels.cuda()
                output = model(images)
                loss = F.nll_loss(output, labels)
                test_loss += loss.detach()
                test_auc += yk_metric.vfe_accuracy_score(labels.cpu(), output.detach().cpu())
                # print(labels.size())
                # schedule.store_val_result(epoch,i,unNorm(images.cpu().data[0]),labels.cpu().data[0],output.cpu().data[0])
            print('               test acc:%.5f' % (test_auc / (i+1)))
            yk_summary.addTestAUC(test_auc / (i+1), epoch)

            best_state['epoch'] = epoch
            best_state['state'] = model.state_dict()
            model_for_save.append(best_state)
    yk_summary.summaryEnd()
    torch.save(model_for_save, filename1)

def load_single_subject_MI(data_folder, subject_id):

    ival = [-500, 4000]
    low_cut_hz = 0
    high_cut_hz = 38
    factor_new = 1e-3
    init_block_size = 1000

    train_filename = 'A{:02d}T.gdf'.format(subject_id)
    test_filename = 'A{:02d}E.gdf'.format(subject_id)
    train_filepath = os.path.join(data_folder, train_filename)
    test_filepath = os.path.join(data_folder, test_filename)
    train_label_filepath = train_filepath.replace('.gdf', '.mat')
    test_label_filepath = test_filepath.replace('.gdf', '.mat')
    print(train_filepath, train_label_filepath)
    print(test_filepath, test_label_filepath)

    train_loader = BCICompetition4Set2A(train_filepath, labels_filename=train_label_filepath)
    test_loader = BCICompetition4Set2A(test_filepath, labels_filename=test_label_filepath)
    train_cnt = train_loader.load()
    test_cnt = test_loader.load()

    # Preprocessing
    train_cnt = train_cnt.drop_channels(['EOG-left', 'EOG-central', 'EOG-right', 'STI 014'])
    assert len(train_cnt.ch_names) == 22
    # lets convert to millvolt for numerical stability of next operations
    train_cnt = mne_apply(lambda a: a * 1e6, train_cnt)
    train_cnt = mne_apply(lambda a: bandpass_cnt(a, low_cut_hz, high_cut_hz, train_cnt.info['sfreq'],filt_order=3,axis=1), train_cnt)
    train_cnt = mne_apply(lambda a: exponential_running_standardize(a.T, factor_new=factor_new,init_block_size=init_block_size,eps=1e-4).T,train_cnt)

    test_cnt = test_cnt.drop_channels(['EOG-left','EOG-central', 'EOG-right', 'STI 014'])
    assert len(test_cnt.ch_names) == 22
    test_cnt = mne_apply(lambda a: a * 1e6, test_cnt)
    test_cnt = mne_apply(lambda a: bandpass_cnt(a, low_cut_hz, high_cut_hz, test_cnt.info['sfreq'],filt_order=3,axis=1), test_cnt)
    test_cnt = mne_apply(lambda a: exponential_running_standardize(a.T, factor_new=factor_new,init_block_size=init_block_size,eps=1e-4).T,test_cnt)
    marker_def = OrderedDict([('Left Hand', [1]), ('Right Hand', [2],),('Foot', [3]), ('Tongue', [4])])

    train_set = create_signal_target_from_raw_mne(train_cnt, marker_def, ival)
    test_set = create_signal_target_from_raw_mne(test_cnt, marker_def, ival)

    data_set = concatenate_two_sets(train_set, test_set)

    return train_set, test_set

def load_multi_subject_MI(data_folder, subject_id_list):
    all_train_set = SignalAndTarget(np.ndarray((0, 22, 1125), dtype='float32'), np.ndarray((0), dtype=int))
    all_test_set = SignalAndTarget(np.ndarray((0, 22, 1125), dtype='float32'), np.ndarray((0), dtype=int))
    for i in subject_id_list:
        subject_id = i+1
        train_set, test_set = load_single_subject_MI(data_folder=data_folder, subject_id=subject_id)
        all_train_set = concatenate_two_sets(all_train_set, train_set)
        all_test_set = concatenate_two_sets(all_test_set, test_set)
    return all_train_set, all_test_set

if __name__ == '__main__':

    print("start...")

    model = EEGNetv4(in_chans=22, n_classes=4, input_time_length=1125, final_conv_length='auto').create_network()
    # yk_kfold_train_MI_across_subject(model1=model, trainname='baseline_eegnet4_MI_KF', logfold='20190701')
    yk_kfold_train_MI_specific_subject(model1=model, trainname='baseline_eegnet4_MI', logfold='20190701')
    #
    # model = ShallowFBCSPNet(in_chans=22, n_classes=4, input_time_length=1125, final_conv_length='auto').create_network()
    # yk_kfold_train_MI_across_subject(model1=model,trainname='baseline_shallowfbcsp_MI_KF', logfold='20190701')
    # yk_kfold_train_MI_specific_subject(model1=model,trainname='baseline_shallowfbcsp_MI', logfold='20190701')
    #
    # # model = Deep4Net(in_chans=22, n_classes=4, input_time_length=1125, final_conv_length='auto',filter_length_4=4).create_network()
    # model = Deep4Net(in_chans=22, n_classes=4, input_time_length=1125, final_conv_length='auto').create_network()
    # yk_kfold_train_MI_across_subject(model1=model,trainname='baseline_deep4_MI_KF', logfold='20190701')
    # yk_kfold_train_MI_specific_subject(model1=model,trainname='baseline_deep4_MI', logfold='20190701')

