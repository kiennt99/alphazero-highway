import tensorflow as tf
from tensorflow import keras

tf.keras.backend.set_floatx('float64')

HIDDEN_STATES = 64
GAME_ACTIONS = 5

'''
We will use 2 Neural Networks for the algorithm implementation.
Note that can be also implemented as one single network sharing the same weights that will produce two outputs.
Also, often there is the usage of a CNN (Convolutional Neural Network) architecture in order to deal with the dynamic pixels of the game directly.
'''

class PolicyV(keras.Model):
    
    '''
    The Value Neural Network will approximate the Value of the node, given a State of the game.
    '''
    
    def __init__(self, input_dim=5):
        
        super(PolicyV, self).__init__()

        self.dense1 = keras.layers.Dense(HIDDEN_STATES, 
                                         activation='relu',
                                         kernel_initializer=keras.initializers.he_normal(),
                                         name='dense_1')

        self.dense2 = keras.layers.Dense(HIDDEN_STATES, 
                                         activation='relu',
                                         kernel_initializer=keras.initializers.he_normal(),
                                         name='dense_2')
        
        self.v_out = keras.layers.Dense(1,
                                        kernel_initializer=keras.initializers.he_normal(),
                                        name='v_out')
        
        
    def call(self, input):               
        
        x = self.dense1(input)
        x = self.dense2(x)       
        x = self.v_out(x)
        
        return x
    
    
    
class PolicyP(keras.Model):
    
    '''
    The Policy Neural Network will approximate the MCTS policy for the choice of nodes, given a State of the game.
    '''
    
    def __init__(self, game_actions=5):
        
        super(PolicyP, self).__init__()

        self.dense1 = keras.layers.Dense(HIDDEN_STATES, 
                                         activation='relu',
                                         kernel_initializer=keras.initializers.he_normal(),
                                         name='dense_1')

        self.dense2 = keras.layers.Dense(HIDDEN_STATES, 
                                         activation='relu',
                                         kernel_initializer=keras.initializers.he_normal(),
                                         name='dense_2')

        self.p_out = keras.layers.Dense(game_actions,
                                        activation='softmax',
                                        kernel_initializer=keras.initializers.he_normal(),
                                        name='p_out')
        
        
    def call(self, input):
        
        x = self.dense1(input)
        x = self.dense2(x)       
        x = self.p_out(x)
        
        return x