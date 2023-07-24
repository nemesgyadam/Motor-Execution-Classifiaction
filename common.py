import  numpy as np


class CircBuff:
    def __init__(self, size, dim=1, dtype=np.float32, ordered=True) -> None:
        self.size = size
        self.dim = dim
        self.ordered = ordered
        self.buffer = np.zeros((size, dim), dtype=dtype)
        self.tail = self.count = 0

    def add(self, data: np.ndarray):
        while len(data.shape) < 2:
            data = np.expand_dims(data, axis=0)
        dsize = data.shape[0]

        start = self.tail
        end = (self.tail + dsize) % self.size
        if start < end:
            self.buffer[start:end] = data
        else:
            self.buffer[start:] = data[:self.size - start]
            self.buffer[:end] = data[self.size - start:]

        self.count = min(self.size, self.count + dsize)
        self.tail = end

    def get(self):
        if self.count < self.size or not self.ordered:
            ret = self.buffer[:self.count]
        else:
            ret = np.concatenate([self.buffer[self.tail:], self.buffer[:self.tail]])

        if self.dim == 1:
            ret = np.squeeze(ret, axis=-1)
        return ret
