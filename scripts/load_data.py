import numpy as np


img_dir = "/Users/macbook/Desktop/PSTAT194/spotted_dogfish/data/"

#Read images and Create training & test DataFrames for transfer learning
#train_sub = readImages(img_dir + "subset") #<class 'pyspark.sql.dataframe.DataFrame'>
#labels_df = pd.read_csv(img_dir + "labels.csv")
#test_df = readImages(img_dir + "test")
#ss = pd.read_csv(img_dir + "sample_submission.csv")


original_train_dir = img_dir + "train"
original_test_dir = img_dir + "test"
train_labels = np.loadtxt(img_dir + 'labels.csv', delimiter=',', dtype=str, skiprows=1)
# Remove missing data, this image was missing on my dataset?
# train_labels = train_labels[train_labels[:, 0] != '000bec180eb18c7604dcecc8fe0dba07']
clazzes, counts = np.unique(train_labels[:, 1], return_counts=True)
print("Some classes with count:")
print(np.asarray((clazzes, counts)).T[0:10])
print("Number of class: %d" % clazzes.size)

import os, shutil

def mkdirIfNotExist(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)
    return directory

base_dir = mkdirIfNotExist('./data_gen')
train_dir = mkdirIfNotExist(os.path.join(base_dir, 'train'))
validation_dir = mkdirIfNotExist(os.path.join(base_dir, 'validation'))
test_dir = mkdirIfNotExist(os.path.join(base_dir, 'test'))
for clazz in clazzes[:]:
    mkdirIfNotExist(os.path.join(train_dir, clazz))
    mkdirIfNotExist(os.path.join(validation_dir, clazz))


def copyIfNotExist(fnames, src_dir, dst_dir):
    nCopied = 0
    for fname in fnames:
        src = os.path.join(src_dir, fname)
        dst = os.path.join(dst_dir, fname)
        if not os.path.exists(dst):
            shutil.copyfile(src, dst)
            nCopied += 1
    if nCopied > 0:
        print("Copied %d to %s" % (nCopied, dst_dir))

# This will split available labeled data to train-validation sets
train_ratio = 0.7
for clazz in clazzes[:]:
    fnames = train_labels[train_labels[:, 1] == clazz][:,0]
    fnames = ['{}.jpg'.format(name) for name in fnames]
    idx = int(len(fnames)*(1-train_ratio))
    val_fnames = fnames[:idx]
    train_fnames = fnames[idx:]
    train_class_dir = os.path.join(train_dir, clazz)
    validation_class_dir = os.path.join(validation_dir, clazz)
    copyIfNotExist(train_fnames, original_train_dir, train_class_dir)
    copyIfNotExist(val_fnames, original_train_dir, validation_class_dir)
