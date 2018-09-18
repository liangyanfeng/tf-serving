
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
import tensorflow as tf


export_dir = '/tmp/test/saved_model/1'
graph_pb = '/Users/liangyanfeng/workspace/serving/tensorflow_serving/example/model.pb'

builder = tf.saved_model.builder.SavedModelBuilder(export_dir)

with tf.gfile.GFile(graph_pb, "rb") as f:
     graph_def = tf.GraphDef()
     graph_def.ParseFromString(f.read())

sigs = {}

with tf.Session(graph=tf.Graph()) as sess:
     tf.import_graph_def(graph_def, name="")
     g = tf.get_default_graph()
     inp = g.get_tensor_by_name('x:0')
     out = g.get_tensor_by_name('y:0')

     sigs['predict_images'] = \
         tf.saved_model.signature_def_utils.predict_signature_def(
             {"images": inp}, {"scores": out})

     builder.add_meta_graph_and_variables(sess,
                                          [tag_constants.SERVING],
                                          signature_def_map=sigs)

builder.save()

'''
print([n.name for n in sess.graph.as_graph_def().node])
frozen_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node_names=['y'])
with open('model.pb', 'wb') as fp:
  fp.write(frozen_graph_def.SerializeToString())
'''

#test(serving/tensorflow_serving/example)
#python mnist_client.py --num_tests=100 --server=localhost:8500
