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
    from keras.layers import Conv2D, MaxPooling2D, Input, Dropout, UpSampling2D, concatenate, ConvLSTM2DCell

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
    lstd1 = GRUConv2D_onepass()(drop4)
    merge6 = concatenate([lstd1, up6], axis=3)
    conv6 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(32, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    lstd2 = ConvLSTM2DCell(32, 3)(conv3)    
    merge7 = concatenate([lstd2, up7], axis=3)
    conv7 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(16, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    lstd3 = ConvLSTM2DCell(16, 3)(conv2)
    merge8 = concatenate([lstd3, up8], axis=3)
    conv8 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(8, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    lstd4 = ConvLSTM2DCell(8,3)(conv1)        
    merge9 = concatenate([lstd4, up9], axis=3)
    conv9 = Conv2D(8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

    #conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    #output = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    #conv9 = Conv2D(5, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    output = Conv2D(5,1, activation=None, padding = 'same')(conv9)

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

def l1_loss(y_true, y_pred):
    from keras import backend as K

    T = K.flatten(y_true)
    P = K.flatten(y_pred)

    error = K.abs(T-P)
    return(K.sum(error))

def weighted_crossentropy(y_true,y_pred,beta=30):
    """
    Assumes y_true and y_pred take on values between [0:1]
    """
    from keras import backend as K
    T = K.flatten(y_true)
    P = K.flatten(y_pred)
    return -20*K.mean(beta*T*K.log(P+1e-3) + (1-T)*K.log(1-P+1e-3))

def loss(y_true, y_pred):
    """
    Assumes y_true = [0:1], y_pred is a logit (can take any value)
    """
    from keras import backend as K
    particle_true = K.flatten(y_true[:,:,:,0])
    P = K.flatten(y_pred[:,:,:,0])
    P = K.clip(P, -70, 70)
    particle_pred = 1/(1 + K.exp(-P))


    loss = weighted_crossentropy(particle_true, particle_pred)
    feature_loss_weight = [1, 1, 1, 1, 5]
    for feature_number in range(1, 5):
        feature_true = K.flatten(y_true[:,:,:,feature_number])
        feature_pred = K.flatten(y_pred[:,:,:,feature_number])

        feature_error = K.abs(feature_true - feature_pred)

        #Add the loss for each pixel which has particle_true = 1, discard those that have particle_true = 0
        feature_loss = K.sum(particle_true * feature_error)/(K.sum(particle_true) + 1e-3)

        loss += feature_loss*feature_loss_weight[feature_number-1]
    return loss

def particle_loss(y_true, y_pred):
    from keras import backend as K
    particle_true = K.flatten(y_true[:,:,:,0])
    P = K.flatten(y_pred[:,:,:,0])
    P = K.clip(P, -70, 70)
    particle_pred = 1/(1 + K.exp(-P))
    return(weighted_crossentropy(particle_true, particle_pred))

def feature_loss(y_true, y_pred, feature_number):
    from keras import backend as K
    particle_true = K.flatten(y_true[:, :, :, 0])
    feature_true = K.flatten(y_true[:, :, :, feature_number])
    feature_pred = K.flatten(y_pred[:, :, :, feature_number])
    feature_error = K.abs(feature_true - feature_pred)
    # Add the loss for each pixel which has particle_true = 1, discard those that have particle_true = 0
    feature_loss = K.sum(particle_true * feature_error) / (K.sum(particle_true) + 1e-3)
    return feature_loss

def particle_binary_accuracy(y_true, y_pred):
    from keras import backend as K
    particle_true = K.flatten(y_true[:,:,:,0])
    P = K.flatten(y_pred[:,:,:,0])
    particle_pred = 1/(1 + K.exp(-P))
    return K.mean(K.equal(particle_true, K.round(particle_pred)))

def x_loss(y_true, y_pred):
    feature_number = 1
    from keras import backend as K
    particle_true = K.flatten(y_true[:, :, :, 0])
    feature_true = K.flatten(y_true[:, :, :, feature_number])
    feature_pred = K.flatten(y_pred[:, :, :, feature_number])
    feature_error = K.abs(feature_true - feature_pred)
    # Add the loss for each pixel which has particle_true = 1, discard those that have particle_true = 0
    feature_loss = K.sum(particle_true * feature_error) / (K.sum(particle_true) + 1e-3)
    return feature_loss

def y_loss(y_true, y_pred):
    feature_number = 2
    from keras import backend as K
    particle_true = K.flatten(y_true[:, :, :, 0])
    feature_true = K.flatten(y_true[:, :, :, feature_number])
    feature_pred = K.flatten(y_pred[:, :, :, feature_number])
    feature_error = K.abs(feature_true - feature_pred)
    # Add the loss for each pixel which has particle_true = 1, discard those that have particle_true = 0
    feature_loss = K.sum(particle_true * feature_error) / (K.sum(particle_true) + 1e-3)
    return feature_loss

def r_loss(y_true, y_pred):
    feature_number = 3
    from keras import backend as K
    particle_true = K.flatten(y_true[:, :, :, 0])
    feature_true = K.flatten(y_true[:, :, :, feature_number])
    feature_pred = K.flatten(y_pred[:, :, :, feature_number])
    feature_error = K.abs(feature_true - feature_pred)
    # Add the loss for each pixel which has particle_true = 1, discard those that have particle_true = 0
    feature_loss = K.sum(particle_true * feature_error) / (K.sum(particle_true) + 1e-3)
    return feature_loss

def i_loss(y_true, y_pred):
    feature_number = 4
    from keras import backend as K
    particle_true = K.flatten(y_true[:, :, :, 0])
    feature_true = K.flatten(y_true[:, :, :, feature_number])
    feature_pred = K.flatten(y_pred[:, :, :, feature_number])
    feature_error = K.abs(feature_true - feature_pred)
    # Add the loss for each pixel which has particle_true = 1, discard those that have particle_true = 0
    feature_loss = K.sum(particle_true * feature_error) / (K.sum(particle_true) + 1e-3)
    # Weight of the feature = 5
    return 5*feature_loss


import tensorflow as tf
from keras import backend as K
from keras.layers import Layer

class MyGRUConv2D_onepass(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(MyLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        return K.dot(x, self.kernel)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


# class GRUConv2D_onepass(Can): # inherit the __call__ method
#     def __init__(self,num_in,num_h,*args,**kwargs):
#         Can.__init__(self)
#         # assume input has dimension num_in.
#         self.num_in,self.num_h = num_in, num_h
#         self.wz = Conv2D(num_in+num_h,num_h,usebias=False,*args,**kwargs)
#         self.wr = Conv2D(num_in+num_h,num_h,usebias=False,*args,**kwargs)
#         self.w = Conv2D(num_in+num_h,num_h,usebias=False,*args,**kwargs)
#         self.incan([self.wz,self.wr,self.w])

#     def __call__(self,i):
#         # assume hidden, input is of shape [batch,num_h] and [batch,num_in]
#         hidden = i[0]
#         inp = i[1]
#         wz,wr,w = self.wz,self.wr,self.w
#         dims = tf.rank(inp)
#         c = tf.concat([hidden,inp],axis=dims-1)
#         z = tf.sigmoid(wz(c))
#         r = tf.sigmoid(wr(c))
#         h_c = tf.tanh(w(tf.concat([hidden*r,inp],axis=dims-1)))
#         h_new = (1-z) * hidden + z * h_c
#         return h_new    

class Can:
    import tensorflow as tf
    import numpy as np
    import time

    from .misc import *


    def __init__(self):
        self.subcans = [] # other cans contained
        self.weights = [] # trainable
        self.biases = []
        self.only_weights = []
        self.variables = [] # should save with the weights, but not trainable
        self.updates = [] # update ops, mainly useful for batch norm
        # well, you decide which one to put into

        self.inference = None

    # by making weight, you create trainable variables
    def make_weight(self,shape,name='W', mean=0., stddev=1e-2, initializer=None):
        mean,stddev = [float(k) for k in [mean,stddev]]
        if initializer is None:
            initial = tf.truncated_normal(shape, mean=mean, stddev=stddev)
        else:
            initial = initializer
        w = tf.Variable(initial,name=name)
        self.weights.append(w)
        self.only_weights.append(w)
        return w

    def make_bias(self,shape,name='b', mean=0.):
        mean = float(mean)
        initial = tf.constant(mean, shape=shape)
        b = tf.Variable(initial,name=name)
        self.weights.append(b)
        self.biases.append(b)
        return b

    # make a variable that is not trainable, by passing in a numpy array
    def make_variable(self,nparray,name='v'):
        v = tf.Variable(nparray,name=name)
        self.variables.append(v)
        return v

    # add an op as update op of this can
    def make_update(self,op):
        self.updates.append(op)
        return op

    # put other cans inside this can, as subcans
    def incan(self,c):
        if hasattr(c,'__iter__'): # if iterable
            self.subcans += list(c)
        else:
            self.subcans += [c]
        # return self

    # another name for incan
    def add(self,c):
        self.incan(c)
        return c

    # if you don't wanna specify the __call__ function manually,
    # you may chain up all the subcans to make one:
    def chain(self):
        def call(i):
            for c in self.subcans:
                i = c(i)
            return i
        self.set_function(call)

    # traverse the tree of all subcans,
    # and extract a flattened list of certain attributes.
    # the attribute itself should be a list, such as 'weights'.
    # f is the transformer function, applied to every entry
    def traverse(self,target='weights',f=lambda x:x):
        l = [f(a) for a in getattr(self,target)] + [c.traverse(target,f) for c in self.subcans]
        # the flatten logic is a little bit dirty
        return list(flatten(l, lambda x:isinstance(x,list)))

    # return weight tensors of current can and it's subcans
    def get_weights(self):
        return self.traverse('weights')

    def get_biases(self):
        return self.traverse('biases')

    def get_only_weights(self): # dont get biases
        return self.traverse('only_weights')

    # return update operations of current can and it's subcans
    def get_updates(self):
        return self.traverse('updates')

    # set __call__ function
    def set_function(self,func):
        self.func = func

    # default __call__
    def __call__(self,i,*args,**kwargs):
        if hasattr(self,'func'):
            return self.func(i,*args,**kwargs)
        else:
            raise NameError('You didnt override __call__(), nor called set_function()/chain()')

    def get_value_of(self,tensors):
        sess = get_session()
        values = sess.run(tensors)
        return values

    def save_weights(self,filename): # save both weights and variables
        with open(filename,'wb') as f:
            # extract all weights in one go:
            w = self.get_value_of(self.get_weights()+self.traverse('variables'))
            print(len(w),'weights (and variables) obtained.')

            # create an array object and put all the arrays into it.
            # otherwise np.asanyarray() within np.savez_compressed()
            # might make stupid mistakes
            arrobj = np.empty([len(w)],dtype='object') # array object
            for i in range(len(w)):
                arrobj[i] = w[i]

            np.savez_compressed(f,w=arrobj)
            print('successfully saved to',filename)
            return True

    def load_weights(self,filename):
        with open(filename,'rb') as f:
            loaded_w = np.load(f)
            print('successfully loaded from',filename)
            if hasattr(loaded_w,'items'):
                # compressed npz (newer)
                loaded_w = loaded_w['w']
            else:
                # npy (older)
                pass
            # but we cannot assign all those weights in one go...
            model_w = self.get_weights()+self.traverse('variables')
            if len(loaded_w)!=len(model_w):
                raise NameError('number of weights (variables) from the file({}) differ from the model({}).'.format(len(loaded_w),len(model_w)))
            else:
                assign_ops = [tf.assign(model_w[i],loaded_w[i])
                    for i,_ in enumerate(model_w)]

            sess = get_session()
            sess.run(assign_ops)
            print(len(loaded_w),'weights assigned.')
            return True

    def infer(self,i):
        # run function, return value
        if self.inference is None:
            # the inference graph will be created when you infer for the first time
            # 1. create placeholders with same dimensions as the input
            if isinstance(i,list): # if Can accept more than one input
                x = [tf.placeholder(tf.float32,shape=[None for _ in range(len(j.shape))])
                    for j in i]
                print('(infer) input is list.')
            else:
                x = tf.placeholder(tf.float32, shape=[None for _ in range(len(i.shape))])

            # 2. set training state to false, construct the graph
            set_training_state(False)
            y = self.__call__(x)
            set_training_state(True)

            # 3. create the inference function
            def inference(k):
                sess = get_session()
                if isinstance(i,list):
                    res = sess.run([y],feed_dict={x[j]:k[j]
                        for j,_ in enumerate(x)})[0]
                else:
                    res = sess.run([y],feed_dict={x:k})[0]
                return res
            self.inference = inference

        return self.inference(i)

    def summary(self):
        print('-------------------')
        print('Directly Trainable:')
        variables_summary(self.get_weights())
        print('-------------------')
        print('Not Directly Trainable:')
        variables_summary(self.traverse('variables'))
        print('-------------------')

def variables_summary(var_list):
    shapes = [v.get_shape() for v in var_list]
    shape_lists = [s.as_list() for s in shapes]
    shape_lists = list(map(lambda x:''.join(map(lambda x:'{:>5}'.format(x),x)),shape_lists))

    num_elements = [s.num_elements() for s in shapes]
    total_num_of_variables = sum(num_elements)
    names = [v.name for v in var_list]

    print('counting variables...')
    for i in range(len(shapes)):
        print('{:>25}  ->  {:<6} {}'.format(
        shape_lists[i],num_elements[i],names[i]))

    print('{:>25}  ->  {:<6} {}'.format(
    'tensors: '+str(len(shapes)),
    str(total_num_of_variables),
    'variables'))