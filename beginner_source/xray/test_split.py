import pandas as pd
import numpy as np
from  sklearn.model_selection import KFold
LABEL_MAP = ['N','A']

def split_set(csv, nsplit = 4):
    pd_frame = pd.read_csv(csv, sep=';')
    file_name = pd_frame.filename.to_numpy()
    label = pd_frame.normal_o_anormal.to_numpy()
    # remove NaN rows
    keep = label != 'nan'
    file_name = file_name[keep]
    label = label[keep]
    # convert string label to numeric label
    assert len(np.unique(label)) == len(LABEL_MAP)
    for i, l in enumerate(LABEL_MAP):
        label[label == l] = i
    label = label.astype(np.int)
    unique, counts = np.unique(label, return_counts=True)
    print(np.asarray((unique, counts)).T)
    # build 4 fold
    # random seed for reproducibility
    kf = KFold(n_splits=nsplit, shuffle=True, random_state=20)
    for train_index, test_index in kf.split(file_name):
        yield file_name[train_index], label[train_index],file_name[test_index], label[test_index]

if __name__ == "__main__":

    for i,(train_file_names,train_labels,test_file_names,test_labels) in enumerate(split_set("/data2/qilei_chen/DATA/xray/xray_dataset_annotations.csv")):
        print(train_labels)    