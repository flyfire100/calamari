import tensorflow as tf
import numpy as np
import json
from typing import Generator, Optional, Any, Dict, Type, List, Tuple

from tfaip.base.model import ModelBase, GraphBase, ModelBaseParams

from calamari_ocr.ocr.backends.model_interface import ModelInterface, NetworkPredictionResult
from calamari_ocr.ocr.callbacks import TrainingCallback
from calamari_ocr.proto.params import ModelParams, LayerType, LayerParams
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import ctc_ops as ctc
from .callbacks.visualize import VisCallback
from .callbacks.earlystopping import EarlyStoppingCallback

keras = tf.keras
K = keras.backend
KL = keras.layers
Model = keras.Model



def calculate_padding(input, scaling_factor):
    def scale(i, f):
        return (f - i % f) % f

    shape = tf.shape(input=input)
    px = scale(tf.gather(shape, 1), scaling_factor[0])
    py = scale(tf.gather(shape, 2), scaling_factor[1])
    return px, py


def pad(input_tensors):
    input, padding = input_tensors[0], input_tensors[1]
    px, py = padding
    shape = tf.keras.backend.shape(input)
    output = tf.image.pad_to_bounding_box(input, 0, 0, tf.keras.backend.gather(shape, 1) + px,
                                          tf.keras.backend.gather(shape, 2) + py)
    return output


class CalamariGraph(GraphBase):
    @classmethod
    def params_cls(cls):
        return ModelParams

    def __init__(self, params: ModelParams):
        super(CalamariGraph, self).__init__(params)

        self.conv_layers: List[Tuple[LayerParams, tf.keras.layers.Layer]] = []
        self.lstm_layers: List[Tuple[LayerParams, tf.keras.layers.Layer]] = []
        cnn_idx = 0
        for layer_index, layer in enumerate([l for l in params.layers if l.type != LayerType.LSTM]):
            if layer.type == LayerType.Convolutional:
                self.conv_layers.append((layer, KL.Conv2D(
                    name="conv2d_{}".format(cnn_idx),
                    filters=layer.filters,
                    kernel_size=(layer.kernel_size.x, layer.kernel_size.y),
                    padding="same",
                    activation="relu",
                )))
                cnn_idx += 1
            elif layer.type == LayerType.Concat:
                self.conv_layers.append((layer, KL.Concatenate(axis=-1)))
            elif layer.type == LayerType.DilatedBlock:
                depth = max(1, layer.dilated_depth)
                dilated_layers = [
                    KL.Conv2D(
                        name='conv2d_dilated{}_{}'.format(i, cnn_idx),
                        filters=layer.filters // depth,
                        kernel_size=(layer.kernel_size.x, layer.kernel_size.y),
                        padding="same",
                        activation="relu",
                        dilation_rate=2 ** (i + 1),
                    )
                    for i in range(depth)
                ]
                concat_layer = KL.Concatenate(axis=-1)
                cnn_idx += 1
                self.conv_layers.append((layer, (dilated_layers, concat_layer)))
            elif layer.type == LayerType.TransposedConv:
                self.conv_layers.append((layer, KL.Conv2DTranspose(
                    name="tconv2d_{}".format(cnn_idx),
                    filters=layer.filters,
                    kernel_size=(layer.kernel_size.x, layer.kernel_size.y),
                    strides=(layer.stride.x, layer.stride.y),
                    padding="same",
                    activation="relu",
                )))
                cnn_idx += 1
            elif layer.type == LayerType.MaxPooling:
                self.conv_layers.append((layer, KL.MaxPool2D(
                    name="pool2d_{}".format(layer_index),
                    pool_size=(layer.kernel_size.x, layer.kernel_size.y),
                    strides=(layer.stride.x, layer.stride.y),
                    padding="same",
                )))
            else:
                raise Exception("Unknown layer of type %s" % layer.type)

        for layer_index, layer in enumerate([l for l in params.layers if l.type == LayerType.LSTM]):
            self.lstm_layers.append((layer, KL.Bidirectional(KL.LSTM(
                units=layer.hidden_nodes,
                activation='tanh',
                recurrent_activation='sigmoid',
                recurrent_dropout=0,
                unroll=False,
                use_bias=True,
                return_sequences=True,
                unit_forget_bias=True,
            ),
                merge_mode='concat',
            )))

        self.dropout = KL.Dropout(params.dropout)
        self.logits = KL.Dense(params.classes, name='logits')
        self.softmax = KL.Softmax(name='softmax')

        def decoded(lgts, output_seq_len):
            return ctc.ctc_greedy_decoder(inputs=array_ops.transpose(lgts, perm=[1, 0, 2]),
                                          sequence_length=tf.cast(K.flatten(output_seq_len),
                                                                  'int32'))[0][0]

        self.sparse_decoded = KL.Lambda(lambda x: decoded(x[0], x[1]), name='sparse_decoded')

    def call(self, inputs, **kwargs):
        params: ModelParams = self._params
        input_data = inputs['img']
        input_sequence_length = K.flatten(inputs['img_len'])
        shape = input_sequence_length, -1

        last_num_filters = 1

        # if concat or conv_T layers are present, we need to pad the input to ensure that possible upsampling layers work properly
        has_concat = any([l.type == LayerType.Concat or l.type == LayerType.TransposedConv for l in params.layers])
        if has_concat:
            sx, sy = 1, 1
            for layer_index, layer in enumerate(
                    [l for l in params.layers if l.type == LayerType.MaxPooling]):
                sx *= layer.stride.x
                sy *= layer.stride.y

            padding = KL.Lambda(lambda x: calculate_padding(x, (sx, sy)), name='compute_padding')(input_data)
            padded = KL.Lambda(pad, name='padded_input')([input_data, padding])
            last_layer_output = padded
        else:
            last_layer_output = input_data

        layers_by_index = []
        for (lp, layer) in self.conv_layers:
            layers_by_index.append(last_layer_output)
            if lp.type == LayerType.Convolutional:
                last_layer_output = layer(last_layer_output)
                last_num_filters = layer.filters
            elif lp.type == LayerType.Concat:
                last_layer_output = layer([layers_by_index[i] for i in layer.concat_indices])
            elif lp.type == LayerType.DilatedBlock:
                depth = max(1, layer.dilated_depth)
                dilated_layers, concat_layer = layer
                dilated_layers = [dl(last_layer_output) for dl in dilated_layers]
                last_layer_output = concat_layer(dilated_layers)
                last_num_filters = (lp.filters // depth) * depth
            elif lp.type == LayerType.TransposedConv:
                last_layer_output = layer(last_layer_output)
                last_num_filters = lp.filters
            elif lp.type == LayerType.MaxPooling:
                last_layer_output = layer(last_layer_output)
                shape = (shape[0] // lp.stride.x, shape[1] // lp.stride.y)
            else:
                raise Exception("Unknown layer of type %s" % lp.type)

        lstm_seq_len, lstm_num_features = shape
        lstm_seq_len = K.cast(lstm_seq_len, 'int32')
        ds = K.shape(last_layer_output)
        last_layer_output = K.reshape(last_layer_output, (ds[0], ds[1], ds[2] * ds[3]))

        # lstm_num_features = last_num_filters * lstm_num_features

        if len(self.lstm_layers) > 0:
            for lstm_params, lstm_layer in self.lstm_layers:
                last_layer_output = lstm_layer(last_layer_output)

        if params.dropout > 0:
            last_layer_output = self.dropout(last_layer_output)

        logits = self.logits(last_layer_output)
        softmax = self.softmax(logits)
        sparse_decoded: tf.SparseTensor = self.sparse_decoded((logits, lstm_seq_len))

        return {
            'out_len': lstm_seq_len,
            'logits': logits,
            'softmax': softmax,
            'sparse_decoded_indices': sparse_decoded.indices,
            'sparse_decoded_values': sparse_decoded.values,
            'sparse_decoded_dense_shape': sparse_decoded.dense_shape,
        }


class CalamariModel(ModelBase):
    @staticmethod
    def get_params_cls() -> Type[ModelBaseParams]:
        return ModelParams

    def create_graph(self, params: ModelBaseParams) -> 'GraphBase':
        return CalamariGraph(params)

    def _loss(self, inputs: Dict[str, tf.Tensor], outputs: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        def to_2d_list(x):
            return K.expand_dims(K.flatten(x), axis=-1)
        loss = KL.Lambda(lambda args: K.ctc_batch_cost(*args), name='ctc')((inputs['gt'], outputs['softmax'],
                                                                            to_2d_list(outputs['out_len']),
                                                                            to_2d_list(inputs['gt_len'])))
        return {
            'loss': loss
        }

    def _extended_metric(self, inputs: Dict[str, tf.Tensor], outputs: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        def create_cer(sparse_decoded, targets, targets_length):
            sparse_decoded = tf.SparseTensor(*sparse_decoded)
            sparse_targets = tf.cast(K.ctc_label_dense_to_sparse(targets, math_ops.cast(
                K.flatten(targets_length), dtype='int32')), 'int32')
            return tf.edit_distance(tf.cast(sparse_decoded, tf.int32), sparse_targets, normalize=True)

        # Note for codec change: the codec size is derived upon creation, therefore the ctc ops must be created
        # using the true codec size (the W/B-Matrix may change its shape however during loading/codec change
        # to match the true codec size
        sparse_decoded = (outputs['sparse_decoded_indices'], outputs['sparse_decoded_values'], outputs['sparse_decoded_dense_shape'])
        cer = KL.Lambda(lambda args: create_cer(*args), output_shape=(1,), name='cer')((sparse_decoded, inputs['gt'], inputs['gt_len']))
        return {
            'CER': cer,
        }

    def __init__(self, params: ModelParams):
        super(CalamariModel, self).__init__(params)
        self._params: ModelParams = params

    def create_solver(self):
        if self.network_proto.solver == NetworkParams.MOMENTUM_SOLVER:
            optimizer = keras.optimizers.SGD(self.network_proto.learning_rate, self.network_proto.momentum, clipnorm=self.network_proto.clipping_norm)
        elif self.network_proto.solver == NetworkParams.ADAM_SOLVER:
            optimizer = keras.optimizers.Adam(self.network_proto.learning_rate, clipnorm=self.network_proto.clipping_norm)
        else:
            raise Exception("Unknown solver of type '%s'" % self.network_proto.solver)

        def ctc_loss(t, p):
            return p

        model = Model(inputs=[self.targets, self.input_data, self.input_length, self.targets_length], outputs=[loss])
        model.compile(optimizer=optimizer, loss={'ctc': ctc_loss},
                      )

        return model

    def load_weights(self, filepath):
        self.model.load_weights(filepath + '.h5')

    def copy_weights_from_model(self, model, indices_to_delete, indices_to_add):
        for target_layer, source_layer in zip(self.model.conv_layers, model.model.conv_layers):
            target_weights = target_layer.weights
            source_weights = source_layer.weights
            if len(target_weights) != len(source_weights):
                raise Exception("Different network structure detected.")

            if len(target_weights) == 0:
                continue

            if target_layer.name.startswith('logits'):
                tW, sW = [(tw, sw) for tw, sw in zip(target_weights, source_weights) if 'kernel' in tw.name][0]
                tB, sB = [(tw, sw) for tw, sw in zip(target_weights, source_weights) if 'bias' in tw.name][0]

                W_val = np.delete(sW.value(), [i - 1 for i in indices_to_delete], axis=1)
                B_val = np.delete(sB.value(), [i - 1 for i in indices_to_delete], axis=0)

                # add new indices at the end
                if list(range(W_val.shape[1], W_val.shape[1] + len(indices_to_add))) != list(sorted(indices_to_add)):
                    raise Exception("Additional labels must be added at the end, but got label indices {} != {}".format(
                        range(W_val.shape[1], W_val.shape[1] + len(indices_to_add)), sorted(indices_to_add)))

                W_val = np.concatenate(
                    (W_val[:, :-1], np.random.uniform(-0.1, 0.1, (W_val.shape[0], len(indices_to_add))), W_val[:, -1:]),
                    axis=1)
                B_val = np.concatenate((B_val[:-1], np.zeros((len(indices_to_add),)), B_val[-1:]), axis=0)

                # reassign values
                tW.assign(W_val)
                tB.assign(B_val)
            else:
                for tw, sw in zip(target_weights, source_weights):
                    tw.assign(sw)

    def train(self, dataset: tf.data.Dataset, validation_dataset: Optional[tf.data.Dataset], checkpoint_params, text_post_proc, progress_bar,
              training_callback=TrainingCallback()):
        predict_func = K.function({t.op.name: t for t in [self.input_data, self.input_length, self.input_params, self.targets, self.targets_length]}, [self.cer, self.sparse_targets, self.sparse_decoded])
        # TODO: correct steps_per_epoch
        # steps_per_epoch = max(1, int(dataset.epoch_size() / checkpoint_params.batch_size))
        # val_s_p_e = 0 if not validation_dataset else max(1, int(np.ceil(validation_dataset.epoch_size() / checkpoint_params.batch_size))),
        steps_per_epoch = 100
        val_s_p_e = 0 if not validation_dataset else 10
        v_cb = VisCallback(training_callback, self.codec, dataset, predict_func, checkpoint_params, steps_per_epoch, text_post_proc)
        es_cb = EarlyStoppingCallback(training_callback, self.codec, validation_dataset, predict_func, checkpoint_params,
                                      val_s_p_e,
                                      steps_per_epoch, v_cb, progress_bar)

        self.model.fit(
            dataset,
            steps_per_epoch=steps_per_epoch,
            epochs=1000,
            use_multiprocessing=False,
            shuffle=False,
            verbose=0,
            callbacks=[
                v_cb, es_cb
            ]
        )

    def predict_raw_batch(self, x: np.array, len_x: np.array) -> Generator[NetworkPredictionResult, None, None]:
        out = self.model.predict_on_batch(
            [tf.convert_to_tensor(x / 255.0, dtype=tf.float32),
             tf.convert_to_tensor(len_x, dtype=tf.int32),
             tf.zeros((len(x), 1), dtype=tf.string)],
        )
        for sm, params, sl in zip(*out):
            sl = sl[0]
            sm = np.roll(sm, 1, axis=1)
            decoded = self.ctc_decoder.decode(sm[:sl])
            pred = NetworkPredictionResult(softmax=sm,
                                           output_length=sl,
                                           decoded=decoded,
                                           )
            yield pred

    def predict_dataset(self, dataset) -> Generator[NetworkPredictionResult, None, None]:
        dataset_gen = self.create_dataset_inputs(dataset, self.batch_size, self.network_proto.features, self.network_proto.backend.shuffle_buffer_size,
                                                 mode='test')
        out = sum([list(zip(self.predict_raw_batch(d[0]['input_data'], d[0]['input_sequence_length']), d[0]['input_data_params'])) for d in dataset_gen], [])
        for pred, params in out:
            enc_param = params[0].numpy()
            pred.params = json.loads(enc_param.decode("utf-8") if isinstance(enc_param, bytes) else enc_param)
            yield pred

    def output_to_input_position(self, x):
        return x * self.scale_factor
