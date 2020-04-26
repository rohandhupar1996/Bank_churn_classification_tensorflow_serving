import tensorflow as tf

class BaseModel(object):
        
    def __init__(self):
        
        self.weight_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.25)
        self.bias_initializer=tf.zeros_initializer()
        
    
    def init_parameters(self):
        '''Initializes the weights and biases of the network '''
        
        with tf.name_scope('weights'):
            self.W1=tf.get_variable('W1', shape=[18,50], dtype=tf.float32, initializer=self.weight_initializer)
            self.W2=tf.get_variable('W2', shape=[50,50], dtype=tf.float32, initializer=self.weight_initializer)
            self.W3=tf.get_variable('W3', shape=[50,2], dtype=tf.float32, initializer=self.weight_initializer)
        
        with tf.name_scope('biases'):
            self.b1=tf.get_variable('bias1', shape=[50], initializer=self.bias_initializer)
            self.b2=tf.get_variable('bias2', shape=[50], initializer=self.bias_initializer)
    
    def forward_propagation(self, x):
        '''Feedforward propagation, computes the output of the neural network.
        
        @param x: the features from the dataset as input for the NN
        @return logits: the logits of the neural network
        @return prediction_scores: the probability scores for the different classes
        '''
        
        with tf.name_scope('feed_forward_propagation'):
            
            z1=tf.matmul(x, self.W1, name='matmul_1')+self.b1
            h1=tf.nn.relu(z1, name='relu_1')
            z2=tf.matmul(h1, self.W2, name='matmul_2')+self.b2
            h2=tf.nn.relu(z2, name='relu_1')
            logits=tf.matmul(h2,self.W3, name='matmul_3')
        
        prediction_scores=tf.nn.softmax(logits)

        return logits, prediction_scores
    
    
    
    
    
    
        