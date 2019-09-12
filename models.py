import tensorflow as tf

def get_lineal_model(num_class):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(32,32,3)))
    model.add(tf.keras.layers.Dense(num_class, activation='softmax'))
    return model

def get_letnet_model(num_class):
    model = tf.keras.Sequential()
    model.add(
        tf.keras.layers.Conv2D(
            filters=6, kernel_size=(5,5), activation='tanh',
            padding='same', input_shape=(32, 32, 3)
        )
    )
    model.add(tf.keras.layers.AveragePooling2D(pool_size=(2,2)))
    model.add(tf.keras.layers.Conv2D(16, (5,5), activation='tanh'))
    model.add(tf.keras.layers.AveragePooling2D((2,2)))
    model.add(tf.keras.layers.Conv2D(120, (5,5), activation='tanh'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(84, activation='tanh'))
    model.add(tf.keras.layers.Dense(num_class, activation='softmax'))
    return model

def get_alex_net(num_class):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(filters=96, input_shape=(224,224,3), kernel_size=(11,11), strides=(4,4), padding= 'valid'))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='valid'))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    model.add(tf.keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(4096, input_shape=(224*224*3,)))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.Dense(4096))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.Dense(num_class))
    model.add(tf.keras.layers.Activation('softmax'))
    return(model)
    
def getVGG(num_class):
    model = tf.keras.Sequential()
    # capa 1 good
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=(224,224,3),strides=1,padding="same"))
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu',padding="same"))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2),strides=2))
    #capa 2
    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu',padding="same"))
    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu',padding="same"))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2),strides=2))
    #capa 3
    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu',padding="same"))
    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu',padding="same"))
    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu',padding="same"))
    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu',padding="same"))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2),strides=2))
    #capa 4
    model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), activation='relu',padding="same"))
    model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), activation='relu',padding="same"))
    model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), activation='relu',padding="same"))
    model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), activation='relu',padding="same"))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2),strides=2))
    #capa 5
    model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), activation='relu',padding="same"))
    model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), activation='relu',padding="same"))
    model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), activation='relu',padding="same"))
    model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), activation='relu',padding="same"))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2),strides=2))
    #flatten
    model.add(tf.keras.layers.Flatten())
    #dense 1 
    model.add(tf.keras.layers.Dense(units=4096,activation="relu"))
    #dense 2
    model.add(tf.keras.layers.Dense(units=4096,activation="relu"))
    #dense 3 
    model.add(tf.keras.layers.Dense(units=1000,activation="relu"))
    #capa final
    model.add(tf.keras.layers.Dense(units=7,activation="softmax"))
    return model

def build_standard_cnn(
    num_filters_per_convolutional_layer,
    num_units_per_dense_layer,
    input_shape,
    num_classes,
    activation='relu',
    maxpool=None):
    """
    """
    model = tf.keras.Sequential()
    model.add(
        tf.keras.layers.Conv2D(
            filters=num_filters_per_convolutional_layer[0],
            kernel_size=(3, 3), activation=activation,
            padding='same', input_shape=input_shape)
        )
    if maxpool != None:
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    for num_filters in num_filters_per_convolutional_layer[1:]:
        if num_filters==0:
            model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        if num_filters==-1:
            model.add(tf.keras.layers.BatchNormalization())        
        else:
            model.add(
                tf.keras.layers.Conv2D(
                    filters=num_filters,
                    kernel_size=(3, 3), activation=activation,
                    padding='same')
            )
        
    model.add(tf.keras.layers.Flatten())
    for num_units in num_units_per_dense_layer:
        if num_units==0:
            model.add(tf.keras.layers.Dropout(0.4))
        else:
            model.add(tf.keras.layers.Dense(num_units, activation=activation))
        
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    
    return model

def identity_block(X, f, filters, stage, block):
    
    # Defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value
    X_shortcut = X
    
    # First component of main path
    X = tf.keras.layers.Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a')(X)
    X = tf.keras.layers.BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = tf.keras.layers.Activation('relu')(X)
    
    # Second component of main path
    X = tf.keras.layers.Conv2D(filters = F2, kernel_size = (f, f), strides = (1, 1), padding = 'same', name = conv_name_base + '2b')(X)
    X = tf.keras.layers.BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = tf.keras.layers.Activation('relu')(X)

    # Third component of main path 
    X = tf.keras.layers.Conv2D(filters = F3, kernel_size = (1, 1), strides = (1, 1), padding = 'valid', name = conv_name_base + '2c')(X)
    X = tf.keras.layers.BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = tf.keras.layers.Add()([X, X_shortcut])
    X = tf.keras.layers.Activation('relu')(X)
    
    return X

def convolutional_block(X, f, filters, stage, block, s=2):

    # Defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value
    X_shortcut = X

    ##### MAIN PATH #####
    # First component of main path 
    X = tf.keras.layers.Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '2a')(X)
    X = tf.keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = tf.keras.layers.Activation('relu')(X)

    # Second component of main path
    X = tf.keras.layers.Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b')(X)
    X = tf.keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = tf.keras.layers.Activation('relu')(X)

    # Third component of main path
    X = tf.keras.layers.Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c')(X)
    X = tf.keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    ##### SHORTCUT PATH #### 
    X_shortcut = tf.keras.layers.Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '1')(X_shortcut)
    X_shortcut = tf.keras.layers.BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = tf.keras.layers.Add()([X, X_shortcut])
    X = tf.keras.layers.Activation('relu')(X)

    return X


def ResNet50(input_shape = (224, 224, 3), classes = 6):
    
    # Define the input as a tensor with shape input_shape
    X_input = tf.keras.layers.Input(input_shape)

    
    # Zero-Padding
    X = tf.keras.layers.ZeroPadding2D((3, 3))(X_input)
    
    # Stage 1
    X = tf.keras.layers.Conv2D(64, (7, 7), strides = (2, 2), name = 'conv1')(X)
    X = tf.keras.layers.BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = tf.keras.layers.Activation('relu')(X)
    X = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(X, f = 3, filters = [64, 64, 256], stage = 2, block='a', s = 1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

    # Stage 3
    X = convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')

    # Stage 4
    X = convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

    # Stage 5
    X = convolutional_block(X, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

    # AVGPOOL
    X = tf.keras.layers.AveragePooling2D(pool_size=(2,2), padding='same')(X)

    # Output layer
    X = tf.keras.layers.Flatten()(X)
    X = tf.keras.layers.Dense(classes, activation='softmax', name='fc' + str(classes))(X)
    
    
    # Create model
    model = tf.keras.Model(inputs = X_input, outputs = X, name='ResNet50')

    return model


def build_standard_cnn2(
    num_filters_per_convolutional_layer,
    num_units_per_dense_layer,
    input_shape,
    num_classes,
    activation='relu',
    ):
    """
    """
    model = tf.keras.Sequential()
    model.add(
        tf.keras.layers.Conv2D(
            filters=num_filters_per_convolutional_layer[0],
            kernel_size=(3, 3), activation=activation,
            padding='same', input_shape=input_shape)
        )    
    for num_filters in num_filters_per_convolutional_layer[1:]:
        if num_filters==0:
            model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        if num_filters==-1:
            model.add(tf.keras.layers.BatchNormalization())        
        else:
            model.add(
                tf.keras.layers.Conv2D(
                    filters=num_filters,
                    kernel_size=(3, 3), activation=activation,
                    padding='same')
            )
        
    model.add(tf.keras.layers.Flatten())
    for num_units in num_units_per_dense_layer:
        if num_units==0:
            model.add(tf.keras.layers.Dropout(0.4))
        else:
            model.add(tf.keras.layers.Dense(num_units, activation=activation))
        
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    
    return model

def standard_cnn(
    num_filters_per_convolutional_layer,
    num_units_per_dense_layer,
    input_shape,
    num_classes,
    activation='relu',
    maxpool=None):
    """
    """
    model = tf.keras.Sequential()
    model.add(
        tf.keras.layers.Conv2D(
            filters=num_filters_per_convolutional_layer[0],
            kernel_size=(3, 3), activation=activation,
            padding='same', input_shape=input_shape)
        )
    if maxpool != None:
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    for num_filters in num_filters_per_convolutional_layer[1:]:
        if num_filters==0:
            model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        if num_filters==-1:
            model.add(tf.keras.layers.BatchNormalization())
        
        else:
            model.add(
                tf.keras.layers.Conv2D(
                    filters=num_filters,
                    kernel_size=(3, 3), activation=activation,
                    padding='same')
            )
        
    model.add(tf.keras.layers.Flatten())
    for num_units in num_units_per_dense_layer:
        if num_units==0:
            model.add(tf.keras.layers.Dropout(0.4))
        else:
            print(num_units)
            model.add(tf.keras.layers.Dense(num_units, activation=activation))
        
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    
    return model