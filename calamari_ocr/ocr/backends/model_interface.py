import numpy as np
from abc import ABC, abstractmethod

from calamari_ocr.ocr.callbacks import TrainingCallback
from .ctc_decoder.ctc_decoder import create_ctc_decoder, CTCDecoderParams
from calamari_ocr.ocr import Codec

from typing import Any, Generator, List


class ModelInterface(ABC):
    def __init__(self, network_proto, graph_type, ctc_decoder_params, batch_size, codec: Codec = None,
                 processes=1):
        """ Interface for a neural net

        Interface above the actual DNN implementation to abstract training and prediction.

        Parameters
        ----------
        network_proto : NetworkParams
            Parameters that define the network
        graph_type : {"train", "test", "deploy"}
            Type of the graph, depending on the type different parts must be added (e.g. the solver)
        ctc_decoder_params :
            Parameters that define the CTC decoder to use
        batch_size : int
            Number of examples to train/predict in parallel
        """
        self.network_proto = network_proto
        self.input_channels = network_proto.channels if network_proto.channels > 0 else 1
        self.graph_type = graph_type
        self.batch_size = batch_size
        self.codec = codec
        self.processes = processes

        self.ctc_decoder = create_ctc_decoder(codec, ctc_decoder_params if ctc_decoder_params else CTCDecoderParams())

    def output_to_input_position(self, x):
        return x

    @abstractmethod
    def train(self, dataset, validation_dataset, checkpoint_params, text_post_proc, progress_bar,
              training_callback=TrainingCallback()):
        pass


    def zero_padding(self, data):
        len_x = [len(x) for x in data]
        out = np.zeros((len(data), max(len_x), self.network_proto.features), dtype=np.uint8)
        for i, x in enumerate(data):
            out[i, 0:len(x)] = x

        return np.expand_dims(out, axis=-1), np.array(len_x, dtype=np.int32)

