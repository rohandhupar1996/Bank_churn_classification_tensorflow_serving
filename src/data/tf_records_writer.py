import os
import tensorflow as tf
import sys
from random import shuffle
import pandas as pd
from preprocess import get_data


def float_feature(value):
    ''' Helper function that wraps float features into the tf.train.Feature class 
    
    @param value: the feature or label of type float, that we want to convert 
    '''
    
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def int64_feature(value):
    ''' Helper function that wraps integer features into the tf.train.Feature class 
    
    @param value: the feature or label of type integer, that we want to convert 
    '''
    
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def write_tf_records_file(x, y, tf_writer):
    '''This function writes a feature-label pair to a TF-Records file
    
    @param x: the feature
    @param y: the label
    @ tf_writer: the TensorFlow Records writer instance that writes the files
    '''
    
    # Convert numpy array to list, because tf.train.Feature accepts only lists
    x=x.tolist()
    
    #Features we want to convert to the binary format
    feature_dict={ 'features': float_feature(x),
                   'labels': float_feature(y),                         
                 }
    
    #Another wrapper class
    features_wrapper=tf.train.Features(feature=feature_dict) 
    # Aaaand another wrapper class lol
    example = tf.train.Example(features=features_wrapper)
    
    # Finally we make the files binary and write them to TF-Records file
    tf_writer.write(example.SerializeToString())



def run(tf_records_dir, data, data_type=None):
    '''Main function for the writing process
    
    @param tf_records_dir: path where the files should be written into
    @param data: the dataset that contains the features and labels
    '''
    
    # If the directory is not present, create one
    try:
        os.stat(tf_records_dir)
    except:
        os.mkdir(tf_records_dir)
      
    #Get the features and labels from the dataset
    features=data[0]
    labels=data[1]
     
    # Number of instances in the dataset
    n_data_instances=features.shape[0]
    # Initialize a counter for the data instances
    data_instance_counter=0
    
    # Specify the number of samples that will be saved in one TF-Records file
    samples_per_file=500
    
    # Number of all TF-Records files in the end
    n_tf_records_files=round(n_data_instances/samples_per_file)
    # Counter for the TF-Records files
    tf_records_counter=0
    
   
    #Iterate over the number of total TF-Records files
    while tf_records_counter < n_tf_records_files:

        # Give each file an unique name(full-path)
        tfrecords_file_name='%s/%s_%i.tfrecord' % (tf_records_dir, data_type, tf_records_counter)

        #Initialize a writer for the files
        with tf.python_io.TFRecordWriter(tfrecords_file_name) as tf_writer:
            
            sample_counter=0

            #Iterate over all data samples and number of samples per TF-Records file
            while data_instance_counter<n_data_instances and sample_counter<samples_per_file:
                
                sys.stdout.write('\r>> Converting data instance %d/%d' % (data_instance_counter+1, n_data_instances))
                sys.stdout.flush()
 
                # Extract a feature instance
                x=features[data_instance_counter]
                # Extract a label instance
                y=labels[data_instance_counter]
                
                # Write feature and label to a TF-Records file
                write_tf_records_file(x,y,tf_writer)

                # Increase the counters
                data_instance_counter+=1
                sample_counter+=1
                
            tf_records_counter+=1
            
    print('\nFinished converting the dataset!')
    
    
if __name__ == "__main__":
    
    # Build the paths for the training, test, and validation TF-Records files
    tf_records_root_dir=os.path.abspath(os.path.join(os.path.dirname("__file__"), '..', 'data/processed/')),
        
    train_dir = os.path.join(tf_records_root_dir, 'tf_records_train')
    test_dir = os.path.join(tf_records_root_dir, 'tf_records_test')
    val_dir = os.path.join(tf_records_root_dir, 'tf_records_val')
    
    # Get the preprocessed data 
    train_data, test_data, val_data=get_data()
    
    #Write for each dataset TF-Records file
    run(train_dir, train_data, data_type='training')
    run(test_dir, test_data, data_type='test')
    run(val_dir, val_data, data_type='validation')
    













