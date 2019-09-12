import matplotlib.pyplot as plt
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image
import os
import uuid
import json
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels


        
def build_sources_from_metadata(metadata, data_dir, mode='train', exclude_labels=None):
    
    if exclude_labels is None:
        exclude_labels = set()
    if isinstance(exclude_labels, (list, tuple)):
        exclude_labels = set(exclude_labels)

    df = metadata.copy()
    df = df[df['split'] == mode]
    df['filepath'] = df['names'].apply(lambda x: os.path.join(data_dir, x))
    include_mask = df['label'].apply(lambda x: x not in exclude_labels)
    df = df[include_mask]

    sources = list(zip(df['filepath'], df['label']))
    return sources
      
def imshow_batch_(batch, show_label=True,cant=3):
    label_batch = batch[1].numpy()
    image_batch = batch[0].numpy()
    fig, axarr = plt.subplots(1, cant, figsize=(15, 5), sharey=True)
    for i in range(cant):
        img = image_batch[i, ...]
        axarr[i].imshow(img)
        if show_label:
            axarr[i].set(xlabel='label = {}'.format(label_batch[i]))

def preprocess_image(image,siz):
    image = tf.image.resize(image, size=(siz,siz))
    image = image / 255.0
    return image

def augment_image(image):
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    image = tf.image.random_flip_left_right(image)
    #image = tf.image.random_flip_up_down(image)
    #image = tf.image.random_hue(image, max_delta=0.2)
    #image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    return image

def make_dataset(sources, training=False, batch_size=1,
    num_epochs=1, num_parallel_calls=1, shuffle_buffer_size=None, pixels = 224, target = 1):
    """
    Returns an operation to iterate over the dataset specified in sources
    Args:
        sources (list): A list of (filepath, label_id) pairs.
        training (bool): whether to apply certain processing steps
            defined only in training mode (e.g. shuffle).
        batch_size (int): number of elements the resulting tensor
            should have.
        num_epochs (int): Number of epochs to repeat the dataset.
        num_parallel_calls (int): Number of parallel calls to use in
            map operations.
        shuffle_buffer_size (int): Number of elements from this dataset
            from which the new dataset will sample.
        pixels (int): Size of the image after resize 
    Returns:
        A tf.data.Dataset object. It will return a tuple images of shape
        [N, H, W, CH] and labels shape [N, 1].
    """
    def load(row):
        filepath = row['image']
        img = tf.io.read_file(filepath)
        img = tf.io.decode_jpeg(img)
        return img, row['label']

    if shuffle_buffer_size is None:
        shuffle_buffer_size = batch_size*4

    images, labels = zip(*sources)
    
    ds = tf.data.Dataset.from_tensor_slices({
        'image': list(images), 'label': list(labels)}) 

    if training:
        ds = ds.shuffle(shuffle_buffer_size)
    
    ds = ds.map(load, num_parallel_calls=num_parallel_calls)
    ds = ds.map(lambda x,y: (preprocess_image(x, pixels), y))
    
    if training:
        ds = ds.map(lambda x,y: (augment_image(x), y))
        
    ds = ds.map(lambda x, y: (x, tuple([y]*target) if target > 1 else y))
    ds = ds.repeat(count=num_epochs)
    ds = ds.batch(batch_size=batch_size)
    ds = ds.prefetch(1)

    return ds
def imshow_with_predictions(model, batch, show_label=True):
    label_batch = batch[1].numpy()
    image_batch = batch[0].numpy()
    pred_batch = model.predict(image_batch)
    fig, axarr = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for i in range(3):
        img = image_batch[i, ...]
        axarr[i].imshow(img)
        pred = int(np.argmax(pred_batch[i]))
        msg = f'pred = {pred}'
        if show_label:
            msg += f', label = {label_batch[i]}'
        axarr[i].set(xlabel=msg)     
        
        
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax       
        
    
    

