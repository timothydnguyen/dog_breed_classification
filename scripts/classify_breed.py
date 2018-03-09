# Dependencies ----
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os.path import join
from sklearn.metrics import log_loss, accuracy_score
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input, decode_predictions
import pickle
import urllib
from tqdm import tqdm

# Global constants ----
INPUT_SIZE = 224
NUM_CLASSES = 16
SEED = 420
POOLING = 'avg'
np.random.seed(seed=SEED)
data_dir = '/Users/timmy/Documents/school/year-3/pstat194/final_project/spotted_dogfish_dev/data/'

labels = pd.read_csv(join(data_dir, 'labels.csv'))
selected_breed_list = list(labels.groupby('breed').count().sort_values(by='id', ascending=False).head(NUM_CLASSES).index)

vgg_bottleneck = VGG16(weights='imagenet', include_top=False, pooling=POOLING)
vgg_bottleneck.compile('sgd','mse')
logreg = pickle.load(open('../models/logreg.sav', 'rb'))

# def read_img(img_id, train_or_test, size):
#     """Read and resize image.
#     # Arguments
#         img_id: string
#         train_or_test: string 'train' or 'test'.
#         size: resize the original image.
#     # Returns
#         Image as numpy array.
#     """
#     img = image.load_img(join(data_dir, train_or_test, '%s.jpg' % img_id), target_size=size)
#     img = image.img_to_array(img)
#     return img
#
#
# labels = labels[labels['breed'].isin(selected_breed_list)]
# labels['target'] = 1
# labels['rank'] = labels.groupby('breed').rank()['id']
# labels_pivot = labels.pivot('id', 'breed', 'target').reset_index().fillna(0)
# np.random.seed(seed=SEED)
# rnd = np.random.random(len(labels))
# train_idx = rnd < 0.8
# valid_idx = rnd >= 0.8
# y_train = labels_pivot[selected_breed_list].values
# ytr = y_train[train_idx]
# yv = y_train[valid_idx]
# x_train = np.zeros((len(labels), INPUT_SIZE, INPUT_SIZE, 3), dtype='float32')
# for i, img_id in tqdm(enumerate(labels['id'])):
#     img = read_img(img_id, 'train', (INPUT_SIZE, INPUT_SIZE))
#     x = preprocess_input(np.expand_dims(img.copy(), axis=0))
#     x_train[i] = x
#
# Xtr = x_train[train_idx]
# Xv = x_train[valid_idx]
# print((Xtr.shape, Xv.shape, ytr.shape, yv.shape))
# valid_vgg_bf = vgg_bottleneck.predict(Xv, batch_size=32, verbose=1)
# valid_probs = logreg.predict_proba(valid_vgg_bf)
# valid_preds = logreg.predict(valid_vgg_bf)
#
# y = (yv * range(NUM_CLASSES)).sum(axis=1)


# Functions ----
def predict_breed(url):
    with urllib.request.urlopen(url) as f:
        img = image.load_img(f, target_size=(INPUT_SIZE, INPUT_SIZE))
    img = image.img_to_array(img)
    # plt.subplot(1, 2, 2)
    # plt.imshow(img/255)
    preprocess_input(np.expand_dims(img.copy(), axis=0))
    img_arr = np.array([img])
    probs = logreg.predict_proba(vgg_bottleneck.predict(img_arr, batch_size=1, verbose=1))
    df = pd.DataFrame({'breed':selected_breed_list, 'probability':probs[0]}).sort_values(by = 'probability', ascending = False)
    print(df)
    prediction = df.breed.iloc[0]
    print('\nIt\'s (probably) a %s!\n' % prediction)
    # plt.show()


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()





# Main ----
if __name__ == '__main__':
    print('\nThis model classifies among the following breeds:')
    print(selected_breed_list)
    print()
    url = input('Enter image URL: ')
    while url != 'q':
        predict_breed(url)
        url = input('Enter image URL: ')
