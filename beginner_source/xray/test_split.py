import pandas as pd
import numpy as np
from  sklearn.model_selection import KFold
import os
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

LABEL_MAP = ['CA','LI','AI','OT']
def split_set_presencia_hallazgos_tb(csv, nsplit = 4):
    pd_frame = pd.read_csv(csv, sep=',')
    file_name = pd_frame.filename.to_numpy()
    label = pd_frame.presencia_hallazgos_tb.to_numpy()
    # remove NaN rows
    keep = label == label
    file_name = file_name[keep]
    label = label[keep]
    temp_file_name = []
    temp_label = []
    count=0
    count2 = 0
    for lb,fn in zip(label,file_name):
        if lb in LABEL_MAP:
            temp_file_name.append(fn)
            temp_label.append(lb)
            command = "cp /data2/qilei_chen/DATA/xray/xray_images/"+fn+" /data2/qilei_chen/DATA/xray/labeled_4categories_images/"+lb
            #print(command)
            #os.system(command)
            if lb=="CA" and os.path.exists("/data2/qilei_chen/DATA/xray/xray_images/"+fn):
                if fn in temp_file_name:
                    count2+=1
                #os.system(command)
                count+=1

    print(count)
    print(count2)
    file_name = np.array(temp_file_name)
    label = np.array(temp_label)
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
    #for train_index, test_index in kf.split(file_name):
    #    yield file_name[train_index], label[train_index],file_name[test_index], label[test_index]


if __name__ == "__main__":
    split_set_presencia_hallazgos_tb("/data2/qilei_chen/DATA/xray/xray_dataset_annotations.csv")
    #for i,(train_file_names,train_labels,test_file_names,test_labels) in enumerate(split_set("/data2/qilei_chen/DATA/xray/xray_dataset_annotations.csv")):
    #    print(train_labels)    