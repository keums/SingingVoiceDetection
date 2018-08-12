from keras.regularizers import l2
from keras.layers.advanced_activations import LeakyReLU
from keras.models import  Model
from keras.layers.core import Dense, Activation
from keras.layers import Input,Conv1D,Conv2D, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization,Dropout,concatenate

def SVAD_CONV_MultiLayer():
    input = Input(shape=(75, 80, 1))
    block1= Conv2D(64, (3, 3), name='conv1_1', padding='same', kernel_initializer='he_normal',kernel_regularizer=l2(0.))(input)
    block1 = BatchNormalization()(block1)
    block1 = LeakyReLU(0.01)(block1)
    block1 = Conv2D(64, (3, 3), name='conv1_2', padding='same', kernel_initializer='he_normal',kernel_regularizer=l2(0.))(block1)
    block1 = BatchNormalization()(block1)
    block1 = LeakyReLU(0.01)(block1)
    block1 = MaxPooling2D((3, 3))(block1)
    block1= Dropout(0.4)(block1)


    block2 = Conv2D(128, (3, 3), name='conv2_1', padding='same', kernel_initializer='he_normal',kernel_regularizer=l2(0.))(block1)
    block2 = BatchNormalization()(block2)
    block2 = LeakyReLU(0.01)(block2)
    block2 = Conv2D(128, (3, 3), name='conv2_2', padding='same', kernel_initializer='he_normal',kernel_regularizer=l2(0.))(block2)
    block2 = BatchNormalization()(block2)
    block2 = LeakyReLU(0.01)(block2)
    block2 = MaxPooling2D((2, 2))(block2)
    block2 = Dropout(0.4)(block2)


    block3 = Conv2D(256, (3, 3), name='conv3_1', padding='same', kernel_initializer='he_normal',kernel_regularizer=l2(0.))(block2)
    block3 = BatchNormalization()(block3)
    block3 = LeakyReLU(0.01)(block3)
    block3 = Conv2D(256, (3, 3), name='conv3_2', padding='same', kernel_initializer='he_normal',kernel_regularizer=l2(0.))(block3)
    block3 = BatchNormalization()(block3)
    block3 = LeakyReLU(0.01)(block3)
    block3 = MaxPooling2D((2, 2))(block3)
    block3 = Dropout(0.4)(block3)


    block4 = Conv2D(256, (3, 3), name='conv4_1', padding='same', kernel_initializer='he_normal',kernel_regularizer=l2(0.))(block3)
    block4 = BatchNormalization()(block4)
    block4 = LeakyReLU(0.01)(block4)
    block4 = Conv2D(256, (3, 3), name='conv4_2', padding='same', kernel_initializer='he_normal',kernel_regularizer=l2(0.))(block4)
    block4 = BatchNormalization()(block4)
    block4 = LeakyReLU(0.01)(block4)
    block4 = MaxPooling2D((2, 2))(block4)
    block4 = Dropout(0.4)(block4)


    max_11 = MaxPooling2D((8, 8))(block1)
    max_22 = MaxPooling2D((4, 4))(block2)
    max_33 = MaxPooling2D((2, 2))(block3)


    lastLayer = concatenate([max_11, max_22,max_33,block4],axis = -1)

    conv1x1 = Conv2D(64, (1, 1), name='conv1x1_1', padding='same', kernel_initializer='he_normal')(lastLayer)
    conv1x1= BatchNormalization()(conv1x1)
    conv1x1 = LeakyReLU(0.01)(conv1x1)
    conv1x1 = Conv2D(1, (1, 1), name='conv1x1_2', padding='same', kernel_initializer='he_normal')(conv1x1)
    conv1x1 = BatchNormalization()(conv1x1)
    conv1x1 = LeakyReLU(0.01)(conv1x1)
    gap = GlobalAveragePooling2D()(conv1x1)
    output = Activation("sigmoid",name="sigmoid")(gap)

    model = Model(inputs = input, outputs = output)
    return model
