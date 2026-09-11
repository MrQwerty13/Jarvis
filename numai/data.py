"""Загрузка изображений MNIST. Здесь нет готовой модели или чужих весов."""

import gzip
import hashlib
import struct
from pathlib import Path
from urllib.request import urlopen

import numpy as np

from .vision import normalize_digit


# Зеркало и контрольные суммы опубликованы в torchvision/datasets/mnist.py.
BASE_URL = 'https://ossci-datasets.s3.amazonaws.com/mnist/'
FILES = {
    'train-images-idx3-ubyte.gz': 'f68b3c2dcbeaaa9fbdd348bbdeb94873',
    'train-labels-idx1-ubyte.gz': 'd53e105ee54ea40749a09fcbcd1e9432',
    't10k-images-idx3-ubyte.gz': '9fb629c4189551a2d022fa330f9573f3',
    't10k-labels-idx1-ubyte.gz': 'ec29112dd5afa0611ce80d1b7f02629c',
}


def download(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, checksum in FILES.items():
        path = directory / name
        if path.exists() and hashlib.md5(path.read_bytes()).hexdigest() == checksum:
            continue
        print('Загрузка ' + name, flush=True)
        with urlopen(BASE_URL + name, timeout=60) as response:
            content = response.read()
        if hashlib.md5(content).hexdigest() != checksum:
            raise ValueError('Не совпала контрольная сумма: ' + name)
        temporary = path.with_suffix('.tmp')
        temporary.write_bytes(content)
        temporary.replace(path)


def read_idx(path):
    with gzip.open(path, 'rb') as source:
        content = source.read()
    if len(content) < 8:
        raise ValueError('Повреждённый IDX: ' + str(path))
    zero, kind, dimensions = struct.unpack('>HBB', content[:4])
    if zero != 0 or kind != 8 or dimensions not in (1, 3):
        raise ValueError('Неизвестный формат IDX: ' + str(path))
    offset = 4 + dimensions * 4
    shape = struct.unpack('>' + 'I' * dimensions, content[4:offset])
    values = np.frombuffer(content, dtype=np.uint8, offset=offset)
    if int(np.prod(shape)) != values.size:
        raise ValueError('Неверная длина IDX: ' + str(path))
    return values.reshape(shape)


def load_split(directory, train=True):
    prefix = 'train' if train else 't10k'
    directory = Path(directory)
    images = read_idx(directory / f'{prefix}-images-idx3-ubyte.gz')
    labels = read_idx(directory / f'{prefix}-labels-idx1-ubyte.gz')
    if images.shape[1:] != (28, 28) or labels.shape != (len(images),) or np.any(labels > 9):
        raise ValueError('Неверные размеры или метки MNIST.')
    normalized = np.stack([normalize_digit(image) for image in images])
    return normalized, labels.astype(np.int64)
