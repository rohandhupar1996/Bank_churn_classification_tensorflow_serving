from models.base_model import BaseModel
import tensorflow as tf


class TrainModel(BaseModel):
    
    def __init__(self, FLAGS):
        
        super(TrainModel,self).__init__()
        
        self.FLAGS=FLAGS
        self.init_parameters()
        
        
    def compute_loss(self, labels, logits):
        '''Compute the cross entropy loss.
        
        @param labels: the labels from the dataset
        @param logits: logits of the neural network
        
        @return loss: the value of the cross entropy
        '''

        with tf.name_scope('loss_function'):
            
            labels=tf.reshape(labels, [self.FLAGS.batch_size])
            cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logits, name='cross_entropy')
            loss=tf.reduce_mean(cross_entropy, name='mean_cross_entropy')
                    
            if self.FLAGS.l2_reg==True:
                l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
            loss = loss +  self.FLAGS.alpha * l2_loss
        
        return loss
    
    
    def accuracy(self, x, labels):
        '''Compute the accuracy of the neural network for a single mini-batch of samples
        
        @param x: the features from the dataset
        @param labels: the labels from the dataset
        
        @return acc: the value of the accuracy
        '''
        
        with tf.name_scope('accuracy'):
            # Get the prediction scores
            _, prediction_scores=self.forward_propagation(x)
            
            # Cast the labels to tf.int64 format
            labels=tf.cast(labels, tf.int64)
            
            # Find out which of the two predicted probability values ​​is higher for two possible classes, 
            # get the index of the class with the higher probability value
            prediction=tf.argmax(prediction_scores, 1)
            
            # Check whether this index (predicted class) corresponds to the label
            is_prediction_correct=tf.equal(prediction, labels)
            
            # Cast the boolean value "is_prediction_correct" to tf.float32, and calculate the mean value (accuracy)
            # accross a mini-batch
            acc=tf.reduce_mean(tf.cast(is_prediction_correct, tf.float32))
        
        return acc
    
    def train_network(self, loss):
        '''Perfrom a gradient descent step.
        
        @param loss: the cross entropy operation node
        
        @return update_op: operation node for the gradient descent
        '''
              
        with tf.name_scope('gradient_descent_operation'):
            
            trainable_variables=tf.trainable_variables() 
            gradients= tf.gradients(loss, trainable_variables, name='compute_gradiends') 
            optimizer=tf.train.AdamOptimizer(self.FLAGS.learning_rate, name='adam_optimizer')
            update_op=optimizer.apply_gradients(zip(gradients, trainable_variables), name='apply_gradients') 

        return update_op
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    