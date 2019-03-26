# Copyright (C) 2019 Yuval Pinter <yuvalpinter@gmail.com>
# NumPy based implemenation of min-heap data structure.
# Adapted from https://github.com/swacad/numpy-heap/
# Changes: debugged A LOT;
#          added minm, minma, maxlen;
#          numpy -> cupy; min -> max;
#          init() accepts array arg

import cupy


class Heap(object):

    def __init__(self, arr=None, length=2):
        self.maxlen = length
        self.heap = cupy.zeros(length)
        self.heap.fill(cupy.nan)
        self.size = 0
        self.minm = float('-inf')
        self.minma = self.maxlen
        if arr is not None:
            self.insert(arr)

    def get_parent_idx(self, child_idx):
        if child_idx == 0:
            return child_idx
        if child_idx % 2 == 0:
            parent_idx = int(child_idx / 2 - 1)
        else:
            parent_idx = int(cupy.floor(child_idx / 2))
        return parent_idx

    def get_parent(self, child_idx):
        parent_idx = self.get_parent_idx(child_idx)
        return self.heap[parent_idx]

    def insert(self, key):
        if type(key) == list or type(key) == cupy.ndarray:
            if key.size == 1:
                key = float(key)
            else:
                for k in key:
                    self.insert(k)
                return

        if key < self.minm:
            return

        if self.size < self.maxlen:
            key_idx = self.size
            self.size += 1
        else:
            key_idx = self.minma
        self.heap[key_idx] = key
        parent_idx = self.get_parent_idx(key_idx)

        # Bubble up until heap property is restored
        while self.heap[key_idx] > self.heap[parent_idx]:
            temp = float(self.heap[parent_idx])
            self.heap[parent_idx] = float(self.heap[key_idx])
            self.heap[key_idx] = temp
            key_idx = parent_idx
            parent_idx = self.get_parent_idx(key_idx)

        if self.size == self.maxlen:
            self.minm = self.heap.min()
            self.minma = self.heap.argmin()

    def get_left_child_idx(self, parent_idx):
        return 2 * parent_idx + 1

    def get_right_child_idx(self, parent_idx):
        return 2 * parent_idx + 2

    def get_left_child(self, parent_idx):
        left_child = self.heap[self.get_left_child_idx(parent_idx)]
        return left_child

    def get_right_child(self, parent_idx):
        right_child = self.heap[self.get_right_child_idx(parent_idx)]
        return right_child

    def get_children_idx(self, parent_idx):
        return 2 * parent_idx + 1, 2 * parent_idx + 2

    def get_children(self, parent_idx):
        child_1_idx, child_2_idx = self.get_parent_idx(parent_idx)
        return self.heap[child_1_idx], self.heap[child_2_idx]

    def extract_max(self):
        root = self.heap[0]
        self.size -= 1
        self.heap[0] = float(self.heap[self.size])
        self.heap[self.size] = cupy.nan

        key_idx = 0
        c1_idx, c2_idx = self.get_children_idx(key_idx)

        # Bubble down root until heap property restored
        ### TODO something here is still not perfect, probably in the last if
        while self.heap[key_idx] < self.heap[c1_idx] or self.heap[key_idx] < self.heap[c2_idx]:
            if self.heap[c1_idx] > self.heap[c2_idx]:
                bigger_child_idx = c1_idx
            else:
                bigger_child_idx = c2_idx
            temp = float(self.heap[bigger_child_idx])
            self.heap[bigger_child_idx] = float(self.heap[key_idx])
            self.heap[key_idx] = temp

            key_idx = bigger_child_idx
            c1_idx, c2_idx = self.get_children_idx(key_idx)
            if c1_idx >= self.size or c2_idx >= self.size:
                break

        return root
    
    def kth(self, k):
        if k >= self.size:
            return self.heap.min()
        for i in range(k):
            if self.size > 0:
                ret = self.extract_max()
        return float(ret)
        