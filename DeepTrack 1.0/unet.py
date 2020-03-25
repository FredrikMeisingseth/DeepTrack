def create_unet(pretrained_weights=None, input_size=(None, None, 1)):
    """Creates a unet that takes inputs of shape (px, px, 1) and outputs of shape (px, px, 5). The output features are:
    1)  Binary, is there a particle here?
    2)  X_vector to center of particle
    3)  Y_vector to center of particle
    4)  Particle radius
    5)  Particle intensity
    The loss function is calculated in the following way:
    1)  Binary crossentropy on the first feature.
    2)  For each pixel, if the first feature label is 1 (there is a particle here), then calculate the L1 loss for the
        remaining features.
    The inputs to the network are images with values within [0:255], the outputs (and labels) are:
    1)      [0:1]
    2-4)    Can take on any values ([0:image_size/2])
    5)      [0:1]
    Inputs:
    pretrained_weights: if not None, loads the pretrained weights into the network
    input_size: the size of the input image (px,px,color channels)
    Outputs:
    network: the created network
    """
    from keras.models import Model
    from keras.optimizers import Adam
    from keras.layers import Conv2D, MaxPooling2D, Input, Dropout, UpSampling2D, concatenate

    input = Input(input_size)

    conv1 = Conv2D(8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(input)
    conv1 = Conv2D(8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(32, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(16, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(8, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

    # conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    # output = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    # conv9 = Conv2D(5, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    output = Conv2D(5, 1, activation=None, padding='same')(conv9)

    model = Model(inputs=[input], outputs=[output])




    model.compile(optimizer=Adam(lr=1e-4), loss=loss, metrics=[particle_loss,
                                                               x_loss,
                                                               y_loss,
                                                               r_loss,
                                                               i_loss,
                                                               particle_binary_accuracy])

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model


def create_multiframe_unet(pretrained_weights=None, input_size=(None, None, None, 1)):
    """Creates a unet that takes inputs of shape (px, px, 1) and outputs of shape (px, px, 5). The output features are:
    1)  Binary, is there a particle here?
    2)  X_vector to center of particle
    3)  Y_vector to center of particle
    4)  Particle radius
    5)  Particle intensity

    The loss function is calculated in the following way:
    1)  Binary crossentropy on the first feature.
    2)  For each pixel, if the first feature label is 1 (there is a particle here), then calculate the L1 loss for the
        remaining features.

    The inputs to the network are images with values within [0:255], the outputs (and labels) are:
    1)      [0:1]
    2-4)    Can take on any values ([0:image_size/2])
    5)      [0:1]

    Inputs:
    pretrained_weights: if not None, loads the pretrained weights into the network
    input_size: the size of the input image (px,px,color channels)

    Outputs:
    network: the created network
    """
    from keras.models import Model
    from keras.optimizers import Adam
    from keras.layers import Conv2D, MaxPooling2D, Input, Dropout, UpSampling2D, concatenate, ConvLSTM2D, \
        TimeDistributed

    input = Input(input_size)

    conv1 = ConvLSTM2D(8, 3, activation='relu', padding='same', kernel_initializer='he_normal', return_sequences=True)(
        input)
    conv1 = TimeDistributed(Conv2D(8, 3, activation='relu', padding='same', kernel_initializer='he_normal'))(conv1)
    pool1 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv1)
    conv2 = ConvLSTM2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal', return_sequences=True)(
        pool1)
    conv2 = TimeDistributed(Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal'))(conv2)
    pool2 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv2)
    conv3 = ConvLSTM2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal', return_sequences=True)(
        pool2)
    conv3 = TimeDistributed(Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal'))(conv3)
    pool3 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv3)
    conv4 = ConvLSTM2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', return_sequences=True)(
        pool3)
    conv4 = TimeDistributed(Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal'))(conv4)
    drop4 = TimeDistributed(Dropout(0.5))(conv4)
    pool4 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(drop4)

    conv5 = TimeDistributed(Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal'))(pool4)
    conv5 = TimeDistributed(Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal'))(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = TimeDistributed(UpSampling2D(size=(2, 2)))(drop5)
    up6 = TimeDistributed(Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal'))(up6)

    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = TimeDistributed(Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal'))(merge6)
    conv6 = TimeDistributed(Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal'))(conv6)

    up7 = TimeDistributed(UpSampling2D(size=(2, 2)))(conv6)
    up7 = TimeDistributed(Conv2D(32, 2, activation='relu', padding='same', kernel_initializer='he_normal'))(up7)

    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = TimeDistributed(Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal'))(merge7)
    conv7 = TimeDistributed(Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal'))(conv7)

    up8 = TimeDistributed(UpSampling2D(size=(2, 2)))(conv7)
    up8 = TimeDistributed(Conv2D(16, 2, activation='relu', padding='same', kernel_initializer='he_normal'))(up8)

    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = TimeDistributed(Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal'))(merge8)
    conv8 = TimeDistributed(Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal'))(conv8)

    up9 = TimeDistributed(UpSampling2D(size=(2, 2)))(conv8)
    up9 = TimeDistributed(Conv2D(8, 2, activation='relu', padding='same', kernel_initializer='he_normal'))(up9)

    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = TimeDistributed(Conv2D(8, 3, activation='relu', padding='same', kernel_initializer='he_normal'))(merge9)
    conv9 = TimeDistributed(Conv2D(8, 3, activation='relu', padding='same', kernel_initializer='he_normal'))(conv9)

    output = TimeDistributed(Conv2D(5, 1, activation=None, padding='same'))(conv9)

    model = Model(inputs=[input], outputs=[output])
    model.compile(optimizer=Adam(lr=1e-4), loss=loss, metrics=[particle_loss,
                                                               x_loss,
                                                               y_loss,
                                                               r_loss,
                                                               i_loss,
                                                               particle_binary_accuracy])

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model


def weighted_crossentropy(y_true, y_pred, beta=45):
    """
    Assumes y_true and y_pred take on values between [0:1]
    """
    from keras import backend as K
    T = K.flatten(y_true)
    P = K.flatten(y_pred)
    return -30 * K.mean(beta * T * K.log(P + 1e-3) + (1 - T) * K.log(1 - P + 1e-3))


def loss(y_true, y_pred):
    """
    Assumes y_true = [0:1], y_pred is a logit (can take any value)
    """
    from keras import backend as K
    particle_true = K.flatten(y_true[:, :, :, 0])
    P = K.flatten(y_pred[:, :, :, 0])
    P = K.clip(P, -70, 70)
    particle_pred = 1 / (1 + K.exp(-P))

    loss = weighted_crossentropy(particle_true, particle_pred)
    feature_loss_weight = [1, 1, 1, 1, 5]
    for feature_number in range(1, 5):
        feature_true = K.flatten(y_true[:, :, :, feature_number])
        feature_pred = K.flatten(y_pred[:, :, :, feature_number])

        feature_error = K.abs(feature_true - feature_pred)

        # Add the loss for each pixel which has particle_true = 1, discard those that have particle_true = 0
        feature_loss = K.sum(particle_true * feature_error) / (K.sum(particle_true) + 1e-3)

        loss += feature_loss * feature_loss_weight[feature_number - 1]
    return loss


def particle_loss(y_true, y_pred):
    from keras import backend as K
    particle_true = K.flatten(y_true[:, :, :, 0])
    P = K.flatten(y_pred[:, :, :, 0])
    P = K.clip(P, -70, 70)
    particle_pred = 1 / (1 + K.exp(-P))
    return (weighted_crossentropy(particle_true, particle_pred))


def particle_binary_accuracy(y_true, y_pred):
    from keras import backend as K
    particle_true = K.flatten(y_true[:, :, :, 0])
    P = K.flatten(y_pred[:, :, :, 0])
    particle_pred = 1 / (1 + K.exp(-P))
    return K.mean(K.equal(particle_true, K.round(particle_pred)))


def x_loss(y_true, y_pred):
    feature_number = 1
    return feature_loss(y_true, y_pred, feature_number)

def y_loss(y_true, y_pred):
    feature_number = 2
    return feature_loss(y_true, y_pred, feature_number)

def r_loss(y_true, y_pred):
    feature_number = 3
    return feature_loss(y_true, y_pred, feature_number)

def i_loss(y_true, y_pred):
    feature_number = 4; feature_weight = 5
    return feature_loss(y_true, y_pred, feature_number, feature_weight)

def feature_loss(y_true, y_pred, feature_number, feature_weight=1):
    from keras import backend as K
    particle_true = K.flatten(y_true[:, :, :, 0])
    feature_true = K.flatten(y_true[:, :, :, feature_number])
    feature_pred = K.flatten(y_pred[:, :, :, feature_number])
    feature_error = K.abs(feature_true - feature_pred)
    # Add the loss for each pixel which has particle_true = 1, discard those that have particle_true = 0
    feature_loss = K.sum(particle_true * feature_error) / (K.sum(particle_true) + 1e-3)
    return feature_weight * feature_loss
