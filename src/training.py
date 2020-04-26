import tensorflow as tf
import numpy as np
import os
from data.dataset import get_training_data, get_test_data, get_validation_data
from models.train_model import TrainModel
from performance import evaluate_model



tf.app.flags.DEFINE_string('train_path', os.path.abspath(os.path.join(os.path.dirname("__file__"), '..', 'data/processed/tf_records_train/')),
                           'Path for the training data.')
tf.app.flags.DEFINE_string('test_path', os.path.abspath(os.path.join(os.path.dirname("__file__"), '..', 'data/processed/tf_records_test/')), 
                           'Path for the test data.')
tf.app.flags.DEFINE_string('validation_path', os.path.abspath(os.path.join(os.path.dirname("__file__"), '..', 'data/processed/tf_records_val/')), 
                           'Path for the test data.')

tf.app.flags.DEFINE_string('checkpoints_path', os.path.abspath(os.path.join(os.path.dirname("__file__"), '..', 'checkpoints/model.ckpt')), 
                           'Path for the test data.')

tf.app.flags.DEFINE_string('train_summary_path', os.path.abspath(os.path.join(os.path.dirname("__file__"), '..', 'summary/train/1')), 
                           'Path for the summaries.'
                           )

tf.app.flags.DEFINE_integer('n_epoch', 1,'Number of epochs.')

tf.app.flags.DEFINE_float('learning_rate', 0.005, 'Learning rate.')

tf.app.flags.DEFINE_boolean('l2_reg', True, 'Whether to use L2 regularization or not.')

tf.app.flags.DEFINE_float('alpha', 1e-4, 'Regularization term.')

tf.app.flags.DEFINE_integer('batch_size', 64,'Batch size.') 

tf.app.flags.DEFINE_integer('eval_after', 10, 'Evaluate performance after number of batches.')

tf.app.flags.DEFINE_integer('n_training_samples', 7000, 'Number of training samples.')

tf.app.flags.DEFINE_integer('n_test_samples', 2000, 'Number of training samples.')

tf.app.flags.DEFINE_integer('n_val_samples', 1000, 'Number of training samples.')

FLAGS = tf.app.flags.FLAGS


def main(_):
    
    n_batches=int(FLAGS.n_training_samples/FLAGS.batch_size)

    # Define the dataflow graph for the training
    training_graph=tf.Graph()
  
    with training_graph.as_default():
        
        writer=tf.summary.FileWriter(FLAGS.train_summary_path)
    
        with tf.name_scope('data_input_pipeline'):
            
            # Access the tf.dataset instance of the tf.data API for the training, 
            # testing and validation of the model
            training_dataset=get_training_data(FLAGS)
            validation_dataset=get_validation_data(FLAGS)
            test_dataset=get_test_data(FLAGS)
            
            # build an interator for each dataset to access the elements of the dataset
            iterator_train = training_dataset.make_initializable_iterator()
            iterator_val = validation_dataset.make_initializable_iterator()
            iterator_test = test_dataset.make_initializable_iterator()
            
            # get the features (x) and labels (y) from the dataset
            x_train, y_train = iterator_train.get_next()
            x_val, y_val = iterator_val.get_next()
            x_test, y_test = iterator_test.get_next()
            
            x_train_copy=tf.identity(x_train,name=None)
            y_train_copy=tf.identity(y_train,name=None)
        
        # Instance of the class containing the functions necessary for the training
        model=TrainModel(FLAGS)
        
        # Build the dataflow graph that represents the neural network for the training
        logits, _ = model.forward_propagation(x_train_copy)
        loss_op=model.compute_loss(y_train_copy, logits) 
        update_op=model.train_network(loss_op)

        acc_train_op=model.accuracy(x_train_copy, y_train_copy)
        acc_val_op=model.accuracy(x_val, y_val)
        
        _, prediction_test=model.forward_propagation(x_test)

        saver=tf.train.Saver()


    # Create a session to execute the datafrom graph
    with tf.Session(graph=training_graph) as sess:
    
        sess.run(tf.global_variables_initializer())
        
        writer.add_graph(sess.graph)

        # Iterate over the number of epochs
        for epoch in range(FLAGS.n_epoch):
  
            sess.run(iterator_train.initializer)
            sess.run(iterator_val.initializer)
            
            temp_loss=0
            temp_acc=0
                       
            print('\n\n\nBegin training... \n')
            # Iterate over the batches
            for i in range(1, n_batches):
                
                _, l, acc_train=sess.run((update_op, loss_op, acc_train_op))
                temp_loss+=l
                temp_acc+=acc_train
                
                # Evaluate the perfromance of the model after "FLAGS.eval_after" 
                # iterations with the validation set              
                if i%FLAGS.eval_after==0:
                
                    acc_val=sess.run(acc_val_op)
              
                    print('epoch_nr: %i, batch_nr: %i/%i, acc_train: %.3f, acc_val: %.3f'
                          %(epoch, i, n_batches, (temp_acc/FLAGS.eval_after), acc_val))
            
                    temp_loss=0
                    temp_acc=0
                       
        
        print('\nSaving Checkpoints...')
        # Save a checkpoint after the training
        saver.save(sess, FLAGS.checkpoints_path)

 
        print('\n\nResult of the evaluation on the test set: \n')
        
        # Test the model after the training is complete with the test dataset
        sess.run(iterator_test.initializer)
        
        # Get the probability scores predicted by the model for each class, 
        # and get also the actual class label
        scores, y_test=sess.run((prediction_test, y_test))    
        
        # Get the class with the highest predicted probability 
        y_pred=np.argmax(scores,axis=1)
        
        # Get the probability values ONLY for the class, where customers leave the bank.
        pos_label_scores=scores[:,1]
        
        # Evaluate the model
        evaluate_model(y_test, y_pred, pos_label_scores)
        
        

    
if __name__ == "__main__":
    tf.app.run()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    