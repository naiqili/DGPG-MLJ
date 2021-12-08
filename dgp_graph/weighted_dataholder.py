import tensorflow as tf
import numpy as np
from numpy.random import RandomState

from gpflow import misc
from gpflow.core.errors import GPflowError
from gpflow.core.compilable import Build
from gpflow.params.parameter import Parameter
from gpflow.params.dataholders import DataHolder

class wMinibatch(DataHolder):

    def __init__(self, value, time_vec, wfunc='exp',
                 batch_size=1, shuffle=True,
                 seed=None, dtype=None, name=None):
        if not misc.is_valid_param_value(value) or misc.is_tensor(value):
            raise ValueError('The value must be either an array or a scalar.')

        super().__init__(value, name=name, dtype=dtype)

        self._batch_size = batch_size
        self._shuffle = shuffle
        self._seed = seed
        
        self.rs = RandomState(seed)
        
        self.time_vec = time_vec
        
        self.n = self._value.shape[0]
        self.wfunc=wfunc
        
        self.update_cur_n(0, 50)
        
        
    def update_cur_n(self, cur_n, cc=50, loc=100):
        self.cur_n = cur_n
        cur_time = self.time_vec[cur_n]
        w = np.zeros((self.n, ))
            
        if self.wfunc == 'exp':
            for i in range(cur_n+1):
                w[i] = (self.time_vec[i]-cur_time) / cc
            for i in range(cur_n+1):
                w[i] = np.exp(w[i])
            w = w / np.sum(w)   
        if self.wfunc == 'logi':
            for i in range(cur_n+1):
                w[i] = (self.time_vec[i] / 50 - cur_n + loc) * cc
            for i in range(cur_n+1):
                w[i] = 1 / (1 + np.exp(-w[i]))
            w = w / np.sum(w)
        if self.wfunc == 'krbf':
            for i in range(cur_n+1):
                w[i] = ((cur_time - self.time_vec[i]) / cc) ** 2
            for i in range(cur_n+1):
                w[i] = np.exp(-w[i])
            w = w / np.sum(w)   
        self.wpdf = w  
        wcdf = w.copy()
        for i in range(1, cur_n+1):
            wcdf[i] += wcdf[i-1]
        self.wcdf = wcdf        

    def gen(self):
        while True:
            r = self.rs.rand()
            for i in range(self.n):
                if self.wcdf[i] > r:
#                     print(i, self._value[i])
                    yield self._value[i]
        
    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        return self.set_batch_size(value)

    @property
    def initializables(self):
        return [self._iterator_tensor]

    @property
    def initializable_feeds(self):
        if self._dataholder_tensor is None:
            return None
        return {self._cache_tensor: self._value,
                self._batch_size_tensor: self._batch_size}

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, seed):
        if self.graph is not None and self.is_built_coherence():
            raise GPflowError('Minibatch seed cannot be changed when it is built.')
        self._seed = seed

    def set_batch_size(self, size, session=None):
        self._batch_size = size
        session = self.enquire_session(session)
        if session is not None:
            self.initialize(session=session, force=True)

    def _clear(self):
        self.reset_name()
        self._cache_tensor = None
        self._batch_size_tensor = None
        self._dataholder_tensor = None
        self._iterator_tensor = None

    def _build(self):
        initial_tensor = self._build_placeholder_cache()
        self._cache_tensor = initial_tensor
        self._dataholder_tensor = self._build_dataholder(initial_tensor)

    def _build_placeholder_cache(self):
        value = self._value
        return tf.placeholder(dtype=value.dtype, shape=None, name='minibatch_init')

    def _build_dataholder(self, initial_tensor):
        if initial_tensor is None:
            raise GPflowError("Minibatch state corrupted.")
        data = tf.data.Dataset.from_tensor_slices(initial_tensor)
        data = data.repeat()
#         if self._shuffle:
#             shape = self._value.shape
#             data = data.shuffle(buffer_size=shape[0], seed=self._seed)
        self._batch_size_tensor = tf.placeholder(tf.int64, shape=())
#         data = data.batch(batch_size=self._batch_size_tensor)
        data = tf.data.Dataset.from_generator(self.gen, output_types=tf.float64).repeat().batch(self._batch_size_tensor)
        self._iterator_tensor = data.make_initializable_iterator()
        name = self._parameter_name()
        return self._iterator_tensor.get_next(name=name)

    def _init_parameter_defaults(self):
        self._cache_tensor = None
        self._batch_size_tensor = None
        self._dataholder_tensor = None
        self._iterator_tensor = None
        self._shuffle = True
        self._batch_size = 1
        self._seed = None

    def _parameter_name(self):
        name = 'minibatch'
        if self.parent is self:
            return misc.tensor_name(self.tf_pathname, name)
        return name
    
    
class wpMinibatch(DataHolder):

    def __init__(self, value, time_vec, wfunc='exp',
                 batch_size=1, shuffle=True,
                 seed=None, dtype=None, name=None):
        if not misc.is_valid_param_value(value) or misc.is_tensor(value):
            raise ValueError('The value must be either an array or a scalar.')

        super().__init__(value, name=name, dtype=dtype)
        
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._seed = seed
        
        self.rs = RandomState(seed)
        
        self.time_vec = time_vec
        self.m = self._value.shape[0]
        self.n = self._value.shape[1]
        self.wfunc=wfunc
        
        self.update_cur_n(0, 50)
        
        
    def update_cur_n(self, cur_n, cc=50, loc=100):
        self.cur_n = cur_n
        cur_time = self.time_vec[cur_n]
        w = np.zeros((self.n, ))
            
        if self.wfunc == 'exp':
            for i in range(cur_n+1):
                w[i] = (self.time_vec[i]-cur_time) / cc
            for i in range(cur_n+1):
                w[i] = np.exp(w[i])
            w = w / np.sum(w)   
        if self.wfunc == 'logi':
            for i in range(cur_n+1):
                w[i] = (self.time_vec[i] / 50 - cur_n + loc) * cc
            for i in range(cur_n+1):
                w[i] = 1 / (1 + np.exp(-w[i]))
            w = w / np.sum(w)
        if self.wfunc == 'krbf':
            for i in range(cur_n+1):
                w[i] = ((cur_time - self.time_vec[i]) / cc) ** 2
            for i in range(cur_n+1):
                w[i] = np.exp(-w[i])
            w = w / np.sum(w)   
        self.wpdf = w  
        wcdf = w.copy()
        for i in range(1, cur_n+1):
            wcdf[i] += wcdf[i-1]
        self.wcdf = wcdf        

    def gen(self):
        while True:
            r = self.rs.rand()
            for i in range(self.n):
                if self.wcdf[i] > r:
                    self.cra = self.rs.randint(self.m)
                    yield self._value[self.cra, i]
        
    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        return self.set_batch_size(value)

    @property
    def initializables(self):
        return [self._iterator_tensor]

    @property
    def initializable_feeds(self):
        if self._dataholder_tensor is None:
            return None
        return {self._cache_tensor: self._value,
                self._batch_size_tensor: self._batch_size}

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, seed):
        if self.graph is not None and self.is_built_coherence():
            raise GPflowError('Minibatch seed cannot be changed when it is built.')
        self._seed = seed

    def set_batch_size(self, size, session=None):
        self._batch_size = size
        session = self.enquire_session(session)
        if session is not None:
            self.initialize(session=session, force=True)

    def _clear(self):
        self.reset_name()
        self._cache_tensor = None
        self._batch_size_tensor = None
        self._dataholder_tensor = None
        self._iterator_tensor = None

    def _build(self):
        initial_tensor = self._build_placeholder_cache()
        self._cache_tensor = initial_tensor
        self._dataholder_tensor = self._build_dataholder(initial_tensor)

    def _build_placeholder_cache(self):
        value = self._value
        return tf.placeholder(dtype=value.dtype, shape=None, name='minibatch_init')

    def _build_dataholder(self, initial_tensor):
        if initial_tensor is None:
            raise GPflowError("Minibatch state corrupted.")
        data = tf.data.Dataset.from_tensor_slices(initial_tensor)
        data = data.repeat()
#         if self._shuffle:
#             shape = self._value.shape
#             data = data.shuffle(buffer_size=shape[0], seed=self._seed)
        self._batch_size_tensor = tf.placeholder(tf.int64, shape=())
#         data = data.batch(batch_size=self._batch_size_tensor)
        data = tf.data.Dataset.from_generator(self.gen, output_types=tf.float64).repeat().batch(self._batch_size_tensor)
        self._iterator_tensor = data.make_initializable_iterator()
        name = self._parameter_name()
        return self._iterator_tensor.get_next(name=name)

    def _init_parameter_defaults(self):
        self._cache_tensor = None
        self._batch_size_tensor = None
        self._dataholder_tensor = None
        self._iterator_tensor = None
        self._shuffle = True
        self._batch_size = 1
        self._seed = None

    def _parameter_name(self):
        name = 'minibatch'
        if self.parent is self:
            return misc.tensor_name(self.tf_pathname, name)
        return name