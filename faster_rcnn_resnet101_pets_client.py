from __future__ import print_function
from __future__ import absolute_import

# Communication to TensorFlow server via gRPC
from grpc.beta import implementations
import tensorflow as tf
import cv2
import time
import numpy as np
from PIL import Image
# TensorFlow serving stuff to send messages
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# Command line arguments
tf.app.flags.DEFINE_string('server', 'localhost:9000',
                       'PredictionService host:port')
tf.app.flags.DEFINE_string('image', '', 'path to image in JPEG format')
FLAGS = tf.app.flags.FLAGS


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
    (im_height, im_width, 3)).astype(np.uint8)


def load_label_map(labels, num_classes):
    label_map = label_map_util.load_labelmap(labels)
    categories = label_map_util.convert_label_map_to_categories(label_map,
							    max_num_classes=num_classes,
							    use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    return category_index


category_index = load_label_map('object_detection/data/pet_label_map.pbtxt', 37)



def main(_):
    host, port = FLAGS.server.split(':')
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)


    # Send request
    start_time = time.time()
    request = predict_pb2.PredictRequest()
    #image = Image.open(FLAGS.image)
    #image_np = load_image_into_numpy_array(image)
    #image_np_expanded = np.expand_dims(image_np, axis=0)

    image = cv2.imread(FLAGS.image)
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image, axis=0)
    print(image_np_expanded.shape)

    # Call model to make prediction on the image
    request.model_spec.name = 'faster_rcnn_resnet101_pets'
    request.model_spec.signature_name = 'serving_default'
    request.inputs['inputs'].CopyFrom(
        tf.contrib.util.make_tensor_proto(image_np_expanded, shape=image_np_expanded.shape, dtype='uint8'))

    result = stub.Predict(request, 60.0)  # 60 secs timeout
    boxes = result.outputs['detection_boxes'].float_val
    classes = result.outputs['detection_classes'].float_val
    scores = result.outputs['detection_scores'].float_val
    print("cost %ss to predict: " % (time.time() - start_time))
    #print(boxes) 
    #print(classes)
    #print(scores)
    vis_util.visualize_boxes_and_labels_on_image_array(
        image,
        np.reshape(boxes,[300,4]),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8)
    
    cv2.namedWindow("detection", cv2.WINDOW_NORMAL)
    cv2.imshow("detection", image)
    cv2.waitKey(0)


if __name__ == '__main__':
    tf.app.run()
