import multiprocessing
from random import shuffle
from time import sleep

import numpy as np
from multiprocessing import Process, Queue

from unet3d.data import DataFile
from unet3d.generator import load_data, DataGenerator


def data_loader(data_filename, subject_ids, queue, skip_blank, sleep_time=0.1, **load_data_kwargs):
    while True:
        _subject_ids = np.copy(subject_ids).tolist()
        shuffle(_subject_ids)
        processes = list()
        while len(_subject_ids) > 0:
            if not queue.full():
                subject_id = _subject_ids.pop()
                process = Process(target=load_data_worker, kwargs=dict(queue=queue, data_filename=data_filename,
                                                                       skip_blank=skip_blank, subject_id=subject_id,
                                                                       **load_data_kwargs))
                process.start()
                processes.append(process)
            else:
                sleep(sleep_time)


def load_data_worker(queue, data_filename, skip_blank=False, **kwargs):
    data_file = DataFile(data_filename, mode='r')
    features, targets = load_data(data_file=data_file, **kwargs)
    if not (skip_blank and np.all(np.equal(targets, 0))):
        queue.put(np.asarray(features), np.asarray(targets))


class MultiProcessingDataGenerator(DataGenerator):
    def __init__(self, sleep_time=0.1, queue_size=12, **kwargs):
        raise RuntimeError("Multiprocessing does not yet work for data generators")
        super(MultiProcessingDataGenerator, self).__init__(**kwargs)
        self.sleep_time = sleep_time
        self.queue = Queue(maxsize=queue_size)
        self.loader_process = None
        self.start_filling_queue()

    def __getitem__(self, index):
        return self.get_batch_from_queue()

    def __iter__(self):
        while True:
            yield self.get_batch_from_queue()

    def start_filling_queue(self):
        self.loader_process = multiprocessing.Process(target=data_loader,
                                                      kwargs=dict(data_filename=self.data_file.filename,
                                                                  subject_ids=self.subject_ids,
                                                                  queue=self.queue,
                                                                  sleep_time=self.sleep_time,
                                                                  skip_blank=self.skip_blank,
                                                                  use_preloaded=self.use_preloaded,
                                                                  normalize=self.normalize,
                                                                  permute=self.permute,
                                                                  translation_deviation=self.translation_deviation,
                                                                  scale_deviation=self.scale_deviation))
        if self.loader_process:
            self.loader_process.start()

    def stop_filling_queue(self):
        if self.loader_process and self.loader_process.is_alive():
            self.loader_process.terminate()

    def get_batch_from_queue(self):
        x = list()
        y = list()
        while len(x) < self.batch_size:
            _x, _y = self.queue.get(timeout=200)
            x.append(_x)
            y.append(_y)
        return np.asarray(x), np.asarray(y)

    def __del__(self):
        self.stop_filling_queue()