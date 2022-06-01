import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
import random
import time

if __name__ == '__main__':

    start = time.process_time()
    print(start)

    FEATRUE = "sex"
    TAG_A = " Male"
    TAG_B = " Female"

    num_samples = 1
    datasets_number = 1     # (TRAIN_POSITIVE_NUM + TRAIN_NEGITIVE_NUM) * datasets_number   0.5  1  3  5  10  20
    k = 10      # iter_number

    TRAIN_POSITIVE_NUM = 9000   # proportion
    TRAIN_NEGITIVE_NUM = 3000
    mutl = "1_3"

    META_BATCH_SIZE = 32
    EPOCHS = 40
    LR = 0.0001
    weightdecay = 0.01

    # Training Dataset
    data_train = pd.read_csv(r'C:/Users/dn/Desktop/data/census-income-new-large.csv')

    # Test Dataset
    data_test = pd.read_csv(r'C:/Users/dn/Desktop/data/census-income-test-new-large.csv')
    x_test, y_test = dataClean(data_test, False)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test),
                                              batch_size=META_BATCH_SIZE)

    models_normal = []
    model = Dedend_Net()

    index = 1
    index_2 = 1

    print(f"{index} th")

    # father model
    model.__init__()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=weightdecay)

    x_train, y_train = dataClean(data_train, True, TRAIN_POSITIVE_NUM, TRAIN_NEGITIVE_NUM)
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train),
                                               batch_size=META_BATCH_SIZE, shuffle=True)
    for epoch in range(1, EPOCHS + 1):
        train(model, optimizer, train_loader)

    end = time.process_time()
    print("pilot model run time : {}".format(end-start))

    for index_2 in range(1, k + 1):   
        print(f"{index}.{index_2} th")

        #  son datasets
        #  Range of values Dataset

        # data_shadow = create_shadow_data(int((TRAIN_POSITIVE_NUM + TRAIN_NEGITIVE_NUM) * datasets_number))
        # x_train_shadow, y_train_shadow = datafilter(data_shadow, model)
        # range_train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train_shadow, y_train_shadow), batch_size=META_BATCH_SIZE, shuffle=True)

        range_train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train),
                                                         batch_size=META_BATCH_SIZE, shuffle=True)

        # son model
        model.__init__()
        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=weightdecay)

        accracy_son_model = []
        for epoch in range(1, EPOCHS + 1):
            acc_temp = train(model, optimizer, range_train_loader)
            accracy_son_model.append(acc_temp)
        end = time.process_time()
        print(end - start)
        print()

    index += 1


    # file = open('datasets/sex/Female_Male_UScensus_models_' + mutl + '_create_1_1_number_' + str(TRAIN_POSITIVE_NUM) + '_' + str(TRAIN_NEGITIVE_NUM) + '_' + str(int(datasets_number * (TRAIN_POSITIVE_NUM + TRAIN_NEGITIVE_NUM))) + '_iter_' + str(k) + '_' + str(FEATRUE), 'wb')
    # cp.dump(models_normal, file)
    # file.close()
    # print("Done")
