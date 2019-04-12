# Copyright (C) 2019 Yuval Pinter <yuvalpinter@gmail.com>
#               2016-2018  Mikel Artetxe <artetxem@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import numpy
from scipy.sparse import csr_matrix as cpu_sm

try:
    import cupy
    from cupy.sparse import csr_matrix as gpu_sm
except ImportError:
    cupy = None
    gpu_sm = None


def supports_cupy():
    return cupy is not None


def get_cupy():
    return cupy


def get_array_module(x):
    if cupy is not None:
        return cupy.get_array_module(x)
    else:
        return numpy


def get_sparse_module(x, dtype='float32', normalize=False):
    if cupy is not None:
        if type(x) == gpu_sm:
            if normalize:
                raise NotImplementedError("can't directly normalize cupy sparse matrix")
            return x
        elif type(x) == cupy.ndarray:
            x /= x.sum()
            return gpu_sm(x)
        if normalize:
            x /= x.sum()
        return gpu_sm(x.astype(dtype))
    else:
        return cpu_sm(x)

        
def asnumpy(x):
    if cupy is not None:
        return cupy.asnumpy(x)
    else:
        return numpy.asarray(x)
