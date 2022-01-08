import logging
import os
import scipy.io as sio
import numpy as np
import sys
import re
import itertools
import mne
from vfe.io.eeglab import read_epochs_from_eeglab_singlefile


def get_file_list_by_lastnum(file_list, num):
    temp_file_list = []
    for filename in file_list:
        str_file_num = filename[filename.find('_', 30) + 1:filename.find('.', 30)]
        if int(str_file_num) == num:
            temp_file_list.append(filename)
    return temp_file_list


def save_data_to_npy():
    filter_nums = 9
    # load data
    s_data = sio.loadmat('./data/subjective_score/subject.mat')
    subject_data = s_data['subjective']  # note that row 4 data is unaviale
    data_root = './data/fbcsp/'
    # read .set file
    file_list = os.listdir(data_root)
    for filter_num in range(filter_nums):
        nums_file_list = get_file_list_by_lastnum(file_list, filter_num+1)
        # build a 0 * 30 (channel) * 375 (time samples)
        X = np.ndarray((0, 30, 375))
        # fatigue label
        Z = np.ndarray((0))
        for filename in nums_file_list:
            file_full_pathname = os.path.join(data_root, filename)
            epoch = read_epochs_from_eeglab_singlefile(file_full_pathname, 'vep_phase')
            # epoch.filter(4,13)
            X = np.concatenate((X, (epoch.get_data() * 1e6).astype(np.float32)))
            # generate fatigue label
            subject_index = int(re.findall(r'[0-9]+', filename)[0])
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

        # saving data
        np.save('save_X_{}'.format(filter_num+1), X)
        np.save('save_Z_{}'.format(filter_num+1), Z)



log = logging.getLogger(__name__)
log.setLevel('INFO')
logging.basicConfig(format='%(asctime)s %(levelname)s : %(message)s', level=logging.INFO, stream=sys.stdout)
# load data
# determinate whether to translate data to numpy data file
# 是否需要重新转换数据为npy文件
hasData = False
if not hasData:
    log.info('translating data to numpy')
    save_data_to_npy()
# starting to load data
# FBCSP个数
fbcsp_filters_nums = 9
n_classes = 3
# ovo多分类组合数
ono_combinations = itertools.combinations(range(n_classes), 2)
# 每个CSP中使用的成分个数
filters_per_band = 30
# 用于序列化CSP滤波器组
models_csp = []
# 用于序列化LDA分类器组
models_classifies = []
# 得到样本个数
temp_x = np.load('save_X_1.npy')
# shuffle样本并切分测试集与训练集
permutation = np.random.permutation(temp_x.shape[0])
index_train = permutation[0:int(0.8*temp_x.shape[0])]
index_test = permutation[int(0.8*temp_x.shape[0]):temp_x.shape[0]]
# 是否需要重新训练模型，针对每对组合均训练一个LDA分类器
needtrain = True
if needtrain:
    # training
    # 每对组合
    for oc in ono_combinations:
        # 每个分类器使用的特征，二维， 样本数 *（filters_per_band * filter个数）
        features = np.zeros((0, filters_per_band))
        # 9个CSP
        for fbcsp_filter_index in range(fbcsp_filters_nums):
            # load 数据
            log.info('loading data for FBCSP {}'.format(fbcsp_filter_index + 1))
            X = np.load('save_X_{}.npy'.format(fbcsp_filter_index + 1))
            Z = np.load('save_Z_{}.npy'.format(fbcsp_filter_index + 1))
            # 样本数 * 30 * 250
            X = X[index_train, :, 75:325]
            Z = Z[index_train]
            log.info('Cal Combinations {}'.format(oc))
            # find all corresponding data
            # 因为此时样本中有3类数据，需要将当前组合中的两类挑出来
            index_class1 = np.where(Z == oc[0])
            index_class2 = np.where(Z == oc[1])
            X_temp_1 = X[index_class1, :, :]
            Z_temp_1 = Z[index_class1]
            X_temp_2 = X[index_class2, :, :]
            Z_temp_2 = Z[index_class2]
            # X_temp_1 变成4维张量，[1,samples,30,250]
            # 合成数据
            epochs_data_train = np.concatenate((X_temp_1, X_temp_2), axis=1)[0]
            if np.shape(features)[0] != 0:
                assert np.mean(np.concatenate((Z_temp_1, Z_temp_2)) == labels) == 1.0
            labels = np.concatenate((Z_temp_1, Z_temp_2))

            # 执行CSP
            from mne.decoding import CSP
            csp = CSP(n_components=filters_per_band, reg=None, log=True, norm_trace=False)
            epochs_data_train_csp_temp = csp.fit_transform(epochs_data_train, labels)
            models_csp.append(csp)
            if np.shape(features)[0] == 0:
                features = np.concatenate((features, epochs_data_train_csp_temp))
            else:
                features = np.concatenate((features, epochs_data_train_csp_temp), axis=1)
            # 存csp和lda的代码没写，需要补的话参考下面pickle。load代码
        # save data
        np.save('save_features_X_train_{}_{}.npy'.format(oc[0], oc[1]), features)
        np.save('save_features_Y_train_{}_{}.npy'.format(oc[0], oc[1]), labels)
        # train
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        lda = LinearDiscriminantAnalysis()
        lda.fit(features, labels)
        models_classifies.append(lda)
else:
    # 如果不重新训练则直接载入模型
    import pickle
    fr = open('model_csp', 'rb')
    models_csp = pickle.load(fr)
    fr.close()
    fr = open('model_lda', 'rb')
    models_classifies = pickle.load(fr)
    fr.close()

# predict
log.info("starting to predict")
count_csp = 0
count_lda = 0
matrix_ovo = np.zeros((len(index_test), n_classes))
ono_combinations = itertools.combinations(range(n_classes), 2)
for oc in ono_combinations:
    features = np.zeros((0, filters_per_band))
    for fbcsp_filter_index in range(fbcsp_filters_nums):
        log.info('loading data for FBCSP {}'.format(fbcsp_filter_index + 1))
        X = np.load('save_X_{}.npy'.format(fbcsp_filter_index + 1))
        Z = np.load('save_Z_{}.npy'.format(fbcsp_filter_index + 1))
        X = X[index_test, :, 75:325]
        Z = Z[index_test]
        epochs_data_train = X
        labels = Z

        # conduct csp
        csp = models_csp[count_csp]
        epochs_data_train_csp_temp = csp.transform(epochs_data_train)
        if np.shape(features)[0] == 0:
            features = np.concatenate((features, epochs_data_train_csp_temp))
        else:
            features = np.concatenate((features, epochs_data_train_csp_temp), axis=1)
        count_csp += 1

    # save test data
    np.save('save_features_X_test_{}_{}.npy'.format(oc[0], oc[1]), features)
    np.save('save_features_Y_test_{}_{}.npy'.format(oc[0], oc[1]), labels)
    lda = models_classifies[count_lda]
    scores = lda.predict(features)
    class_balance = np.mean(labels == labels[0])
    class_balance = max(class_balance, 1. - class_balance)
    print("Classification accuracy: %f / Chance level: %f" % (np.mean(scores == labels), class_balance))
    count_lda += 1

    for i in range(len(index_test)):
        matrix_ovo[int(i), int(scores[i])] += 1
models_labels = []
for i in range(len(index_test)):
    models_labels.append(np.where(matrix_ovo[i, :] == np.max(matrix_ovo[i, :]))[0][0])
print("Classification accuracy: %f" % (np.mean(models_labels == labels)))
