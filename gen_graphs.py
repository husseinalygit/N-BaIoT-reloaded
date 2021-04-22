from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import time
from tensorflow import keras

devices_ids = [1, 2, 4, 5]
data_loc = 'processed'

cms = []
mses = []
trs = []
scores = []
for id in devices_ids:
    print('working on device {}'.format(id))
    training_df = pd.read_csv('{}/{}/{}'.format(data_loc, id, 'training.csv'), index_col=0)
    testing_df = pd.read_csv('{}/{}/{}'.format(data_loc, id, 'testing.csv'), index_col=0)

    start_time = time.time()

    X_train_Val = training_df.loc[:, training_df.columns != 'class'].values

    X_test = testing_df.loc[:, testing_df.columns != 'class'].values
    y_test = testing_df['class'].values

    X_train, X_val = train_test_split(X_train_Val, random_state=1337, test_size=0.2)

    t = MinMaxScaler()
    X_train = t.fit_transform(X_train)
    X_val = t.transform(X_val)
    X_test = t.transform(X_test)

    # load_model
    model = keras.models.load_model('models/autoencoder_{}'.format(id))

    X_val_prediction = model.predict(X_val)
    mse = np.mean(np.power(X_val - X_val_prediction, 2), axis=1)
    mses.append(mse)
    print("mean is %.5f" % mse.mean())
    print("min is %.5f" % mse.min())
    print("max is %.5f" % mse.max())
    print("std is %.5f" % mse.std())

    tr = mse.mean() + mse.std()
    trs.append(tr)

    print("tr is {}".format(tr))
    X_test_prediction = model.predict(X_test)
    testing_mse = np.mean(np.power(X_test - X_test_prediction, 2), axis=1)

    test_labels = np.asarray([0 if testing_mse[i] <= tr else 1 for i in range(len(testing_mse))])
    y_true = np.asarray([0 if label == 0 else 1 for label in y_test])

    cm = confusion_matrix(y_true, test_labels)
    f1 = f1_score(y_true, test_labels)
    p = precision_score(y_true, test_labels)
    r = recall_score(y_true, test_labels)
    scores.append([f1, p, r])
    print(cm)
    cms.append(cm)
    print('Execution took {:.2f}'.format((time.time() - start_time) / 60))

for index, id in enumerate(devices_ids):
    precision = scores[index][1]

    f1 = scores[index][0]
    TP = cms[index][0][0]
    FP = cms[index][0][1]
    FN = cms[index][1][0]
    TN = cms[index][1][1]

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP / (TP + FN)
    # Specificity or true negative rate
    TNR = TN / (TN + FP)
    # Precision or positive predictive value
    PPV = TP / (TP + FP)
    # Negative predictive value
    NPV = TN / (TN + FN)
    # Fall out or false positive rate
    FPR = FP / (FP + TN)
    # False negative rate
    FNR = FN / (TP + FN)
    # False discovery rate
    FDR = FP / (TP + FP)

    print(
        "for device {}:\nPrecession: {:.3f} \nf1: {:.3f}\nTPR: {:.3f} \nFPR: {:.3f} \nFNR: {:.3f}\nTNR: {:.3f}".format(
            id, precision, f1, TPR, FPR, FNR, TNR))
