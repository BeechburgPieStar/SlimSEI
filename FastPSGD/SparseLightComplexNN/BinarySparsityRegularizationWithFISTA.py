from keras.engine.topology import Layer
from keras.engine.topology import InputSpec
import keras.backend as K
from keras import initializers
from keras import regularizers
from keras.regularizers import l1
import numpy as np


def SoftSign(W, threshold):
    W = np.sign(W)*np.maximum(abs(W)-threshold, 0)
    return W

def HardSign(W, threshold):
    W1 = np.sign(W-threshold)
    W2 = np.sign(W+threshold)
    W = (W1+W2)/2
    return W

scale = 1
mu = 1e-5
class SparsityRegularization(Layer):
    def __init__(self, l1=0.01, threshold=0.5, **kwargs):
        if K.image_dim_ordering() == 'tf':
            self.axis = -1
        else:
            self.axis = 1
        self.l1 = l1
        self.threshold = threshold
        super(SparsityRegularization, self).__init__(**kwargs)

    def build(self, input_shape):
        dim = input_shape[self.axis]
        if dim is None:
            raise ValueError('Axis ' + str(self.axis) + ' of '
                             'input tensor should have a defined dimension '
                             'but the layer received an input with shape ' +
                             str(input_shape) + '.')
        self.input_spec = InputSpec(ndim=len(input_shape),
                                    axes={self.axis: dim})
        self.gamma = self.add_weight(shape=(dim,),
                                     initializer = 'Ones',#initial,
                                     name='gamma',
                                     regularizer=regularizers.get(l1(l=self.l1)),
                                     trainable=True
                                     )

        super(SparsityRegularization, self).build(input_shape)
        Weights = K.get_value(self.weights[0])
        self.t=1
        self.fullPrecisionWeights = Weights# t FP weights
        self.lastIterationfullPrecisionWeights = Weights#t-1 FP weight
        BinaryWeights = HardSign(Weights, self.threshold)
        self.lastIterationWeights = BinaryWeights.copy()# t Bin weights
        K.set_value(self.weights[0], BinaryWeights)


    def on_batch_end(self):
        t = self.t
        Weights = K.get_value(self.weights[0])
        weightsUpdate = Weights - self.lastIterationWeights
        temp = self.fullPrecisionWeights + weightsUpdate + (t-2)/(t+1)*(self.fullPrecisionWeights - self.lastIterationfullPrecisionWeights)#FISTA step1
        temp = SoftSign(temp, mu)#FISTA step2
        temp = np.clip(temp, -1, 1)
        scale = 1/(np.max(abs(temp)))
        newfullPrecisionWeights = temp * scale
        BinaryWeights = HardSign(newfullPrecisionWeights, self.threshold)

        self.lastIterationfullPrecisionWeights = self.fullPrecisionWeights #t-1 FP weight
        self.fullPrecisionWeights = newfullPrecisionWeights# t FP weights      
        self.lastIterationWeights = BinaryWeights.copy()# t Bin weights   
        self.t = t + 1  
        K.set_value(self.weights[0], BinaryWeights)

    def call(self, inputs, mask=None):
        return inputs * self.gamma

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'l1': self.l1
        }
        base_config = super(SparsityRegularization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
