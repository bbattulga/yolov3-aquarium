"""
Retrain the YOLO model for your own dataset.
"""
import keras
from tensorflow.python.lib.io import file_io
import tensorflow as tf
from . import downloads
from keras.utils import plot_model  # plot model
from .yolo3.utils import get_random_data
from .yolo3.model import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input, Lambda
import keras.backend as K
import numpy as np
import argparse
import time

config = tf.ConfigProto(device_count={'GPU': 1, 'CPU': 8})
sess = tf.Session(config=config)
keras.backend.set_session(sess)

if False:
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
else:
    import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()

parser.add_argument("-a", "--annotation_path", type=str,
                    default='test_data/training_data/annotation.txt', help="input annotation_path")
parser.add_argument("-c", "--classes_path", type=str,
                    default='test_data/training_data/pedestrian_classes.txt', help="input classes_path")
parser.add_argument("-o", "--output_model_path", type=str,
                    default='model_data/pedestrian_model.h5', help="input output_model_path")
parser.add_argument("-epochs", "--epochs", type=int,
                    default=30, help="number of epochs")
parser.add_argument("-batch_size", "--batch_size", type=int,
                    default=30, help="train batch size")
parser.add_argument("-freeze", "--freeze", type=int,
                    default=0, help="train on frozen layers")
parser.add_argument(
    '--job-dir',
    dest='job_dir',
    default='',
    type=str,
    help='GCS location to write checkpoints and export models')
args = parser.parse_args()

gcp_save_root = str(args.job_dir)


def visualize(history_dict):
    global gcp_save_root
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.title('Loss')
    plt.plot(history_dict['loss'], label='Training Loss')
    plt.plot(history_dict['val_loss'], label='Validation Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title('Accuracy')
    plt.plot(history_dict['acc'], label='Training Accuracy')
    plt.plot(history_dict['val_acc'], label='Validation Accuracy')
    plt.legend()
    fig_path = './visual.png'
    plt.savefig(fig_path)
    gcp_visual_save_path = gcp_save_root + \
        '/aqua-visual-' + str(time.time()) + '.png'
    with file_io.FileIO(fig_path, mode='rb') as saved_fig:
        with file_io.FileIO(gcp_visual_save_path, mode='wb+') as output:
            output.write(saved_fig.read())
            print('saved to', gcp_visual_save_path)
    try:
        plt.show()
    except:
        pass


def save_model_to_cloud(model):
    global gcp_save_root
    # save model
    gcp_model_save_path = gcp_save_root + \
        '/aquarium-' + str(time.time()) + '.h5'
    model_save_path = './aquarium_model.h5'
    model.save(model_save_path)
    with file_io.FileIO(model_save_path, mode='rb') as saved_model:
        with file_io.FileIO(gcp_model_save_path, mode='wb+') as output:
            output.write(saved_model.read())
            print('saved to ' + gcp_model_save_path)


def _main(annotation_path, classes_path, output_model_path, epochs=30, batch_size=16):
    # return
    annotation_path = annotation_path
    log_dir = 'logs/000/'
    classes_path = classes_path
    anchors_path = './model_data/yolo_anchors.txt'
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)

    input_shape = (416, 416)  # multiple of 32, hw

    is_tiny_version = len(anchors) == 6  # default setting
    if is_tiny_version:
        model = create_tiny_model(input_shape, anchors, num_classes,
                                  freeze_body=2, weights_path='./tiny_yolo_weights.h5')
    else:
        model = create_model(input_shape, anchors, num_classes,
                             freeze_body=2, weights_path='./yolo_weights.h5')  # make sure you know what you freeze
    # model.save('yolo_model_retrain.h5')  # creates a HDF5 file 'my_model.h5'
    # save_model_to_cloud(model)
    # plot_model(model, to_file='./model_data/retrained_model.png',
    #            show_shapes=True)

    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                 monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(
        monitor='val_loss', min_delta=0, patience=10, verbose=1)

    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
        lines = list(map(lambda line: './train/' + line, lines))
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val

    # Train with frozen layers first, to get a stable loss.
    # Adjust num epochs to your dataset. This step is enough to obtain a not bad model.
    if args.freeze:
        # model.compile(optimizer=Adam(lr=1e-3), loss={
        #     # use custom yolo_loss Lambda layer.
        #     'yolo_loss': lambda y_true, y_pred: y_pred})

        model.compile(optimizer=Adam(lr=1e-3),
                      loss='mean_squared_error', metrics=['accuracy'])

        batch_size = args.batch_size
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(
            num_train, num_val, batch_size))
        history = model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
                                      steps_per_epoch=max(
                                          1, num_train//batch_size),
                                      validation_data=data_generator_wrapper(
            lines[num_train:], batch_size, input_shape, anchors, num_classes),
            validation_steps=max(1, num_val//batch_size),
            epochs=epochs,
            initial_epoch=0,
            callbacks=[logging])

        # model.save_weights(log_dir + 'trained_weights_stage_1.h5')
        # model.save(log_dir + 'trained_model_stage_1.h5')
        visualize(history.history)

    # Unfreeze and continue training, to fine-tune.
    # Train longer if the result is not good.
    if not args.freeze:
        for i in range(len(model.layers)):
            model.layers[i].trainable = True
        # recompile to apply the change
        model.compile(optimizer=Adam(lr=1e-4),
                      loss='mean_squared_error', metrics=['accuracy'])
        print('Unfreeze all of the layers.')

        batch_size = 1  # note that more GPU memory is required after unfreezing the body
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(
            num_train, num_val, batch_size))
        history = model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
                                      steps_per_epoch=max(
                                          1, num_train//batch_size),
                                      validation_data=data_generator_wrapper(
            lines[num_train:], batch_size, input_shape, anchors, num_classes),
            validation_steps=max(1, num_val//batch_size),
            epochs=epochs,
            initial_epoch=0,
            callbacks=[logging, reduce_lr, early_stopping])
        # model.save_weights(log_dir + 'trained_weights_final.h5')
        # model.save(log_dir + 'trained_model_final.h5')
        visualize(history.history)
        save_model_to_cloud(model)

    # Further training if needed.

    # print('model.input = ',model.input)
    # print('len(model.layers) = ',len(model.layers))
    # print('model.layers[-1]: ',model.layers[-1].output)
    # print('model.layers[-2]: ',model.layers[-2].output)
    # print('model.layers[-3]: ',model.layers[-3].output)
    # print('model.layers[-4]: ',model.layers[-4].output,'\n')
    # # original yolo model outputs:
    # print('model.layers[-5]: ',model.layers[-5].output)
    # print('model.layers[-6]: ',model.layers[-6].output)
    # print('model.layers[-7]: ',model.layers[-7].output)

    # save the derived model for detection(using yolo_video.py)
    # derived_model = Model(model.input[0], [
    #                       model.layers[249].output, model.layers[250].output, model.layers[251].output])
    # plot_model(derived_model,
    #            to_file=output_model_path[:-3]+'.png', show_shapes=True)
    # derived_model.save(output_model_path)


def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    # with open(anchors_path) as f:
    #     anchors = f.readline()
    anchors = '10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326'
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


def create_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
                 weights_path='./yolo_weights.h5'):
    '''create the training model'''
    K.clear_session()  # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    # y_true = [Input(shape=(416//{0:32, 1:16, 2:8}[l], 416//{0:32, 1:16, 2:8}[l], 9//3, 80+5)) for l in range(3)]
    y_true = [Input(shape=(h//{0: 32, 1: 16, 2: 8}[l], w//{0: 32, 1: 16, 2: 8}[
                    l], num_anchors//3, num_classes+5)) for l in range(3)]

    model_body = yolo_body(image_input, num_anchors//3, num_classes)
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(
        num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze darknet53 body or freeze all but 3 output layers.
            num = (185, len(model_body.layers)-3)[freeze_body-1]
            for i in range(num):
                model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(
                num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
                        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)
    print('model_body.input: ', model_body.input)
    print('model.input: ', model.input)

    return model


def create_tiny_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
                      weights_path='./tiny_yolo_weights.h5'):
    '''create the training model, for Tiny YOLOv3'''
    K.clear_session()  # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h//{0: 32, 1: 16}[l], w//{0: 32, 1: 16}[l],
                           num_anchors//2, num_classes+5)) for l in range(2)]

    model_body = tiny_yolo_body(image_input, num_anchors//2, num_classes)
    print('Create Tiny YOLOv3 model with {} anchors and {} classes.'.format(
        num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze the darknet body or freeze all but 2 output layers.
            num = (20, len(model_body.layers)-2)[freeze_body-1]
            for i in range(num):
                model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(
                num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
                        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.7})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model


def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes):
    '''data generator for fit_generator'''
    n = len(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i == 0:
                np.random.shuffle(annotation_lines)
            image, box = get_random_data(
                annotation_lines[i], input_shape, random=True)
            image_data.append(image)
            box_data.append(box)
            i = (i+1) % n
        image_data = np.array(image_data)   # input of original yolo: image
        box_data = np.array(box_data)       # output of original yolo: boxes
        # some kind of output description?!
        y_true = preprocess_true_boxes(
            box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true], np.zeros(batch_size)


def data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes):
    n = len(annotation_lines)
    if n == 0 or batch_size <= 0:
        return None
    return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes)


if __name__ == '__main__':

    # print('annotation_path = ', args.annotation_path)
    # print('classes_path = ', args.classes_path)
    # print('output_model_path = ', args.output_model_path)
    # print('epochs =', args.epochs)
    # print('train batch size =', args.batch_size)
    annotation_path = './train/_annotations.txt'
    classes_path = './train/_classes.txt'
    output_model_path = './model_output.h5'
    epochs = 30
    batch_size = 32

    _main(annotation_path,
          classes_path,
          output_model_path,
          epochs=epochs,
          batch_size=batch_size)
