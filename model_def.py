#---------------------------------------------
# Implemented by Gustavo Assunção
# Institute of Systems and Robotics, Coimbra
# June 2019
#---------------------------------------------

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Dense, ZeroPadding2D, Activation
from keras.models import Model
from keras.layers import Input

def vggvox_model(input_shape):

	length = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
	dim = [2, 5, 8, 11, 14, 17, 20, 23, 27, 30] 

	idx = length.index(input_shape[1])
	n = dim[idx]

	img_input = Input(input_shape)

	pad_inp = ZeroPadding2D(padding=(1,1),name='pad_1')(img_input)
	x1 = Conv2D(96, (7, 7), strides=(2, 2), padding='valid', activation='relu', use_bias=True, name='conv1')(pad_inp)
	x2 = MaxPooling2D(pool_size=(3, 3), strides=(2,2), padding='valid', name='pool1')(x1)

	pad_x2 = ZeroPadding2D(padding=(1,1),name='pad_2')(x2)
	x3 = Conv2D(256, (5, 5), strides=(2, 2), padding='valid', activation='relu', use_bias=True, name='conv2')(pad_x2)
	x4 = MaxPooling2D(pool_size=(3, 3), strides=(2,2), padding='valid', name='pool2')(x3)

	pad_x4 = ZeroPadding2D(padding=(1,1),name='pad_3')(x4)
	x5 = Conv2D(384, (3, 3), strides=(1, 1), padding='valid', activation='relu', use_bias=True, name='conv3')(pad_x4)

	pad_x5 = ZeroPadding2D(padding=(1,1),name='pad_4')(x5)
	x6 = Conv2D(256, (3, 3), strides=(1, 1), padding='valid', activation='relu', use_bias=True, name='conv4')(pad_x5)

	pad_x6 = ZeroPadding2D(padding=(1,1),name='pad_5')(x6)
	x7 = Conv2D(256, (3, 3), strides=(1, 1), padding='valid', activation='relu', use_bias=True, name='conv5')(pad_x6)
	x8 = MaxPooling2D(pool_size=(5, 3), strides=(3,2), padding='valid', name='pool5')(x7)

	x9 = Conv2D(4096, (9, 1), strides=(1, 1), padding='valid', activation='relu', use_bias=True, name='fc6')(x8)

	x10 = AveragePooling2D(pool_size=(1, n), strides=(1, 1), padding='valid', name='pool6')(x9)

	x11 = Dense(1024, activation='relu', use_bias=True, name='fc7')(x10)

	score = Dense(1300, use_bias=True, name='fc8')(x11)
	prob = Activation('relu', name='act_out')(score)

	model = Model(img_input, score)
	return model
	




