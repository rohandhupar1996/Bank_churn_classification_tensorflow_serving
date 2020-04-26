import tensorflow as tf
import os
from models.inference_model import InferenceModel

tf.app.flags.DEFINE_string('checkpoints_path', os.path.abspath(os.path.join(os.path.dirname( __file__ ), 
                                                                            '..', 'checkpoints/')), 
                           'Path of the checkpoints.')


tf.app.flags.DEFINE_string('export_path_base', os.path.abspath(os.path.join(os.path.dirname( __file__ ), 
                                                                            '..', 'model-export/')), 
                           'Directory where to export the model.')

tf.app.flags.DEFINE_string('test_path', os.path.abspath(os.path.join(os.path.dirname("__file__"), 
                                                                     '..', 'data/processed/tf_records_test/')), 
                           'Path for the test data.')

tf.app.flags.DEFINE_string('model_name', 'churn_prediction', 'Name of the model.')

tf.app.flags.DEFINE_integer('model_version', 1, 'Version number of the model.')

FLAGS = tf.app.flags.FLAGS

def main(_):
    
    # Build a dataflow graph for the inference. This inference graph contains only a small part
    # of the original dataflow graph for training
    
    inference_graph=tf.Graph()
    
    with inference_graph.as_default():
             
        # Instance of the model class
        model=InferenceModel()

        # Placeholder for data input during inference
        input_data=tf.placeholder(tf.float32, shape=[1, 18])
        
        # Compute the logits and the probability scores for the two possible classes
        logits, prediction_scores=model.forward_propagation(input_data)
        
        saver = tf.train.Saver()
        
    with tf.Session(graph=inference_graph) as sess:
        
        # Restore the learned weights and biases of neural network from the last checkpoint
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoints_path)   
        saver.restore(sess, ckpt.model_checkpoint_path)      
    
        # Build the export path. path/to/model/MODELNAME/VERSION/
        export_path = os.path.join(tf.compat.as_bytes(FLAGS.export_path_base),
                                   tf.compat.as_bytes(FLAGS.model_name),
                                   tf.compat.as_bytes(str(FLAGS.model_version))
                                   )
        

        print('Exporting trained model to %s'%export_path)
        
        # Instance of the build for the SavedModel format
        builder = tf.saved_model.builder.SavedModelBuilder(export_path)
        
        
        # Create tensor info for datainput and output of the neural network
        predict_tensor_inputs_info = tf.saved_model.utils.build_tensor_info(input_data)
        predict_tensor_scores_info = tf.saved_model.utils.build_tensor_info(prediction_scores)
        
            
        # Build prediction signature
        prediction_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                inputs={'inputs': predict_tensor_inputs_info},
                outputs={'probability_scores': predict_tensor_scores_info},
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
            )
        )
         
            
        # Export the model in a SavedModel format
        builder.add_meta_graph_and_variables(
            sess, # session
            [tf.saved_model.tag_constants.SERVING], #tags for the metagraph
            signature_def_map={
                tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:prediction_signature
            })

        builder.save()
        
        

if __name__ == "__main__":
    tf.app.run()





























