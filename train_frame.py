import torch.nn.functional as F
import torch
import tqdm
from vfe.util.EEGDataSet import EEGDATA
from vfe.util.trainSchedule import Scheduler
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
from Nets.MoletNet.MorletNet import MorletNet

def yk_kfold_train_framework_acrosssubject(trainname, logfold, model1, k=5):
    train_name = trainname
    log_fold = logfold
    # load data, EEGDATA 分配比例是训练集、测试集和验证集
    eeg_data = EEGDATA(data_path='./data/npy/save_X.npy', label_path='./data/npy/save_Z.npy', percentage=(0.8, 0.2, 0))
    x_data = eeg_data.get_trainset().get_x().numpy()
    y_data = eeg_data.get_trainset().get_y().numpy()
    testloader = DataLoader(eeg_data.get_testset(), batch_size=64, drop_last=True)
    kf = KFold(n_splits=k)
    i = 1
    for train_index, val_index in kf.split(x_data):
        print('processing fold {}'.format(i))
        x_train_data, x_val_data = x_data[train_index], x_data[val_index]
        y_train_data, y_val_data = y_data[train_index], y_data[val_index]
        traindataset = EEGDataSet(x_train_data, y_train_data)
        valdataset = EEGDataSet(x_val_data, y_val_data)
        train_loader = DataLoader(traindataset, batch_size=64, sampler=traindataset.get_sampler(), drop_last=True)
        val_loader = DataLoader(valdataset, batch_size=64, drop_last=True)
        train_name1 = train_name + "{}".format(i)
        model = copy.deepcopy(model1)
        yk_train(logfoldname=log_fold, trainname=train_name1, modell=model, trainloader=train_loader, valloader=val_loader, testloader=testloader)
        i += 1

def yk_singleSub_train_framework_subject(trainname1, logfold, model1):
    train_name = trainname1
    log_fold = logfold
    # load data
    s_data = sio.loadmat('./data/subjective_score/subject.mat')
    subject_data = s_data['subjective']  # note that row 4 data is unaviale
    data_root = './data/erp/'
    # read .set file
    file_list = os.listdir(data_root)
    kp = LeavePOut(1)
    for j in np.arange(len(file_list)):
        # 临时修正
        if j != 2:
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
            train_name1 = train_name + "{}_KF{}".format(j, i)
            model = copy.deepcopy(model1)
            yk_train(logfoldname=log_fold, trainname=train_name1, modell=model, trainloader=train_loader,
                     valloader=val_loader, testloader=testloader)
            i += 1

def yk_leavek_train_framework_subject(trainname1, logfold, model1, k = 1):
    train_name = trainname1
    log_fold = logfold
    # load data
    s_data = sio.loadmat('./data/subjective_score/subject.mat')
    subject_data = s_data['subjective']  # note that row 4 data is unaviale
    data_root = './data/erp/'
    # read .set file
    file_list = os.listdir(data_root)
    kp = LeavePOut(k)
    i = 1
    for train_index, test_index in kp.split(range(len(file_list))):
        file_train = np.array(file_list)[train_index]
        file_test = np.array(file_list)[test_index]
        train_x, train_y = read_data_from_filelist(dataroot=data_root, file_name_list=file_train)
        val_x, val_y = read_data_from_filelist(dataroot=data_root, file_name_list=file_test)
        traindataset = EEGDATA(X=train_x, Z=train_y, percentage=(1.0,0,0)).get_trainset()
        valdataset = EEGDATA(X=val_x, Z=val_y, percentage=(1.0,0,0)).get_trainset()
        train_loader = DataLoader(traindataset, batch_size=64, sampler=traindataset.get_sampler(), drop_last=True)
        val_loader = DataLoader(valdataset, batch_size=64, drop_last=True)
        train_name1 = train_name + "{}".format(i)
        model = copy.deepcopy(model1)
        yk_train(logfoldname=log_fold, trainname=train_name1, modell=model, trainloader=train_loader, valloader=val_loader, testloader=None)
        i += 1

def yk_train_framework_withdiffdrop():
    train_name = 'MoeletInceptionNet-best'
    log_fold = 'log20190620'
    # load data using eegdata
    eeg_data = EEGDATA(data_path='./data/npy/save_X.npy', label_path='./data/npy/save_Z.npy')
    train_loader = DataLoader(eeg_data.get_trainset(), batch_size=64, sampler=eeg_data.get_trainset().get_sampler(), drop_last=True)
    val_loader = DataLoader(eeg_data.get_valset(), batch_size=64, shuffle=True, drop_last=True)
    test_loader = DataLoader(eeg_data.get_testset(), batch_size=64, shuffle=True, drop_last=True)

    for index, prob in enumerate(zip([0.4, 0.4, 0.4, 0.4, 0.5, 0.5, 0.5,0.5, 0.6, 0.6,0.6,0.6,0.7,0.7,0.7,0.7], [0.4, 0.5, 0.6, 0.7,0.4, 0.5, 0.6, 0.7,0.4, 0.5, 0.6, 0.7,0.4, 0.5, 0.6, 0.7])):
        train_name = train_name + "{}".format(index)
        model = MorletInceptionNet(drop0=prob[0], drop1=prob[1])
        yk_train(logfoldname=log_fold,trainname=train_name, modell=model, trainloader=train_loader, valloader=val_loader, testloader=test_loader)

def yk_train(logfoldname, trainname, modell, trainloader, valloader, testloader,lr0=2*(2e-3)):
    # Parameters
    train_name=trainname
    filename1 = './' + trainname + '.t7'
    # init mode
    model = modell.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=4e-3)
    schedule = Scheduler(lr=lr0, total_epoches=201, lr_decay_rate=2, lr_decay_epoch=30, lr_min=1e-5, eval_epoch=1)
    yk_summary = Summary(train_name, logfoldname)
    yk_metric = Metric()
    # best metrics
    best_state = {
        'state': model.state_dict(),
        'epoch': 0
    }
    model_for_save = list()
    for epoch in range(schedule.get_total_epoches()):
        running_loss = 0.0
        model.train()
        for i, (images, labels) in tqdm.tqdm(enumerate(trainloader)):
            images = images.float().cuda()
            labels = labels.squeeze().cuda()
            optimizer.zero_grad()
            output = model(images)
            loss = F.nll_loss(output, labels)
            loss.backward()
            running_loss += loss.detach()
            optimizer.step()
        print('epoch[%d/%d] train loss:%.5f'%(epoch, schedule.get_total_epoches(), running_loss/(i+1)))
        yk_summary.addTrainLoss(running_loss/(i+1), epoch)
        ##################################lr decay###############################################
        if epoch % schedule.get_decay_epoch() == 0:
            schedule.decay_learning_rate()
            print('current lr is: %.5f' % (schedule.get_learning_rate()))
            optimizer = torch.optim.Adam(model.parameters(), lr=schedule.get_learning_rate(), betas=(0.5, 0.9))
            yk_summary.addLearningRate(schedule.get_learning_rate(), epoch)
    #     #########################################################################################
    #     ##################################eval epoch#############################################
        if epoch % schedule.get_eval_epoch()==0:
            model.eval()
            temp_auc = 0.0
            temp_kappa = 0.0
            eval_loss=0.0
            for i, (images, labels) in tqdm.tqdm(enumerate(valloader)):
                images = images.float().cuda()
                labels = labels.cuda()
                output = model(images)
                loss = F.nll_loss(output, labels)
                eval_loss += loss.detach()
                temp_auc += yk_metric.vfe_accuracy_score(labels.cpu(), output.detach().cpu())
                temp_kappa += yk_metric.vfe_kappa(labels.cpu(), output.detach().cpu())
                # print(labels.size())
                # schedule.store_val_result(epoch,i,unNorm(images.cpu().data[0]),labels.cpu().data[0],output.cpu().data[0])
            print('epoch[%d/%d]  val loss:%.5f' % (epoch, schedule.get_total_epoches(), eval_loss / (i+1)))
            tt = eval_loss/(i+1)
            yk_summary.addValLoss(tt, epoch)
            yk_summary.addValAUC(temp_auc / (i+1), epoch)
            yk_summary.addValKappa(temp_kappa/(i+1), epoch)

            if testloader is None:
                continue
            test_auc = 0.0
            test_loss = 0.0
            for i, (images, labels) in tqdm.tqdm(enumerate(testloader)):
                images = images.float().cuda()
                labels = labels.cuda()
                output = model(images)
                loss = F.nll_loss(output, labels)
                test_loss += loss.detach()
                test_auc += yk_metric.vfe_accuracy_score(labels.cpu(), output.detach().cpu())
                # print(labels.size())
                # schedule.store_val_result(epoch,i,unNorm(images.cpu().data[0]),labels.cpu().data[0],output.cpu().data[0])
            print('epoch[%d/%d]  test auc:%.5f' % (epoch, schedule.get_total_epoches(), test_auc / (i+1)))
            yk_summary.addTestAUC(test_auc / (i+1), epoch)

            best_state['epoch'] = epoch
            best_state['state'] = model.state_dict()
            model_for_save.append(best_state)
    yk_summary.summaryEnd()
    torch.save(model_for_save, filename1)

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


if __name__ == '__main__':
    # yk_kfold_train_framework_eacrosssubject(trainname='morletInceptionNet_test_KF', logfold='20190630', model1=MorletInceptionNet())
    #yk_kfold_train_framework_acrosssubject(trainname='cnnInceptionNet_test_KF', logfold='20190630', model1=CNNInceptionNet())
    yk_singleSub_train_framework_subject(trainname1='morletInceptionNet_test_sub', logfold='20190711', model1=MorletInceptionNet())
    #yk_singleSub_train_framework_subject(trainname1='cnnInceptionNet_test_sub', logfold='20190630', model1=CNNInceptionNet())
    from Nets.CosInceptionNet import CosInceptionNet

    #yk_singleSub_train_framework_subject(trainname1='cosInceptionNet_test_sub', logfold='20190710',  model1=MorletNet())
