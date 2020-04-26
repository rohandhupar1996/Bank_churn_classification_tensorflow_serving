import tensorflow as tf
import os


def get_training_data(FLAGS):
    '''Prepares training dataset with the tf-data API for the input pipeline.
    Reads TensorFlow Records files from the harddrive and applies several
    transformations to the files, like mini-batching, shuffling etc.
    
    @return dataset: the training dataset
    '''
    
    filenames=[FLAGS.train_path+'/'+f for f in os.listdir(FLAGS.train_path)]
    
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(parse)
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.repeat()
    dataset = dataset.batch(FLAGS.batch_size)
    dataset = dataset.prefetch(buffer_size=1)
    
    return dataset
 


def get_validation_data(FLAGS):
    '''Prepares validation dataset with the tf-data API for the input pipeline.
    Reads TensorFlow Records files from the harddrive and applies several
    transformations to the files, like mini-batching, shuffling etc.
    
    @return dataset: the validation dataset
    '''
    
    filenames=[FLAGS.validation_path+'/'+f for f in os.listdir(FLAGS.validation_path)]
    
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(parse)
    dataset = dataset.shuffle(buffer_size=1)
    dataset = dataset.repeat()
    dataset = dataset.batch(FLAGS.n_val_samples)
    dataset = dataset.prefetch(buffer_size=1)
    
    return dataset

def get_test_data(FLAGS):
    '''Prepares test dataset with the tf-data API for the input pipeline.
    Reads TensorFlow Records files from the harddrive and applies several
    transformations to the files, like mini-batching, shuffling etc.
    
    @return dataset: the test dataset
    '''
    
    filenames=[FLAGS.test_path+'/'+f for f in os.listdir(FLAGS.test_path)]
    
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(parse)
    dataset = dataset.shuffle(buffer_size=FLAGS.n_test_samples)
    dataset = dataset.repeat()
    dataset = dataset.batch(FLAGS.n_test_samples)
    dataset = dataset.prefetch(buffer_size=1)
    
    return dataset


def parse(serialized):

    features={'features':tf.FixedLenFeature([18], tf.float32),
              'labels':tf.FixedLenFeature([1], tf.float32),
              }
    
    
    parsed_example=tf.parse_single_example(serialized,
                                           features=features,
                                           )
 
    features=parsed_example['features']
    label = tf.cast(parsed_example['labels'], tf.int32)
    
    return features, label






