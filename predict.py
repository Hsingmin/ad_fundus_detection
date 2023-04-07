
import tensorflow as tf, sys

# runs forward prediction
def predict(image_path):
    # reads in image
    image_data = tf.gfile.FastGFile(image_path, 'rb').read()

    # loads label file, strips off carriage return
    label_lines = [line.rstrip() for line
                       in tf.gfile.GFile("retrained_labels.txt")]

    # unpersists graph from file
    with tf.gfile.FastGFile("retrained_graph.pb", 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

    probs = dict({})

    with tf.Session() as sess:
        # feed the image_data as input to the graph and get first prediction
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

        # predictions
        predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})

        # sorts to show labels of first prediction in order of confidence
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
        
        # Loop through top_k
        for node_id in top_k:
            probs[label_lines[node_id]] = predictions[0][node_id]
            # log[0](log[1], '{} (score = {:.5f})'.format(label_lines[node_id], predictions[0][node_id]))

    return probs
