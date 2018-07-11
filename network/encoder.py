import tensorflow as tf
import numpy as np

class Encoder(object):

    def __init__(self, is_training, drop_prob = None):
        self.is_training = is_training
        self.dropout = drop_prob

    def __call__(self, is_training = False, drop_prob = None):
        self.is_training = is_training
        self.dropout = drop_prob

    def __dropout__(self, x):
        if self.dropout:
            return tf.layers.dropout(x, rate = self.dropout, training = self.is_training)
        else:
            return x

    def __mask__(self, mask):
        mask_embedding = tf.constant([[0,0,0], [1,0,0], [0,1,0], [0,0,1]], dtype=np.float32)
        return tf.nn.embedding_lookup(mask_embedding, mask)

    def __pooling__(self, x, max_length, hidden_size):
        x = tf.reshape(x, [-1, max_length, hidden_size])
        x = tf.reduce_max(x, axis = 1)
        return tf.reshape(x, [-1, hidden_size])

    def __piece_pooling__(self, x, max_length, hidden_size, mask):
        x = tf.reshape(x, [-1, max_length, hidden_size, 1])
        x = tf.reduce_max(tf.reshape(self.__mask__(mask), [-1, max_length, 1, 3]) * 100 + x, axis = 1) - 100
        return tf.reshape(x, [-1, hidden_size * 3])

    def __cnn_cell__(self, x, hidden_size, kernel_size, stride_size):
        x = tf.expand_dims(x, axis=1)
        x = tf.layers.conv2d(inputs=x, 
                             filters = hidden_size, 
                             kernel_size = [1, kernel_size], 
                             strides = [1, stride_size], 
                             padding = 'same', 
                             kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
        return x

    def cnn(self, x, hidden_size, kernel_size = 3, stride_size = 1, activation=tf.nn.relu):
        with tf.name_scope("cnn"):
            max_length = x.get_shape()[1]
            x = self.__cnn_cell__(x, hidden_size, kernel_size, stride_size)
            x = self.__pooling__(x, max_length, hidden_size)
            x = activation(x)
            return self.__dropout__(x)

    def pcnn(self, x, hidden_size, mask, kernel_size = 3, stride_size = 1, activation=tf.nn.relu):
        with tf.name_scope("pcnn"):
            max_length = x.get_shape()[1]
            x = self.__cnn_cell__(x, hidden_size, kernel_size, stride_size)
            x = self.__piece_pooling__(x, max_length, hidden_size, mask)
            x = activation(x)
            return self.__dropout__(x)

    def __rnn_cell__(self, dim, cell_name='lstm'):
        if isinstance(cell_name, list) or isinstance(cell_name, tuple):
            if len(cell_name) == 1:
                return self.__rnn_cell__(dim, cell_name[0])
            cells = [self.__rnn_cell__(dim, c) for c in cell_name]
            return tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
        if cell_name.lower() == 'lstm':
            return tf.contrib.rnn.BasicLSTMCell(dim, state_is_tuple=True)
        elif cell_name.lower() == 'gru':
            return tf.contrib.rnn.GRUCell(dim)
        raise NotImplementedError

    def rnn(self, x, hidden_size, sequence_length, cell_name='lstm'):
        with tf.name_scope('rnn'):
            x = self.__dropout__(x)
            cell = self.__rnn_cell__(hidden_size, cell_name)
            _, states = tf.nn.dynamic_rnn(cell, x, sequence_length = sequence_length, dtype = tf.float32, scope = 'dynamic-rnn')
            if isinstance(states, tuple):
                states = states[0]
            return states

    def birnn(self, x, hidden_size, sequence_length, cell_name='lstm'):
        with tf.name_scope('bi-rnn'):
            x = self.__dropout__(x)
            fw_cell = self.__rnn_cell__(hidden_size, cell_name)
            bw_cell = self.__rnn_cell__(hidden_size, cell_name)
            _, states = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, x, sequence_length = sequence_length, dtype = tf.float32, scope = 'dynamic-bi-rnn')
            fw_states, bw_states = states
            if isinstance(fw_states, tuple):
                fw_states = fw_states[0]
                bw_states = bw_states[0]
            return tf.concat([fw_states, bw_states], axis = 1)

    def __normalize__(self,
                      inputs, 
                      epsilon = 1e-8,
                      scope="ln",
                      reuse=tf.AUTO_REUSE):
        '''Applies layer normalization.
        
        Args:
          inputs: A tensor with 2 or more dimensions, where the first dimension has
            `batch_size`.
          epsilon: A floating number. A very small number for preventing ZeroDivision Error.
          scope: Optional scope for `variable_scope`.
          reuse: Boolean, whether to reuse the weights of a previous layer
            by the same name.
          
        Returns:
          A tensor with the same shape and data dtype as `inputs`.
        '''
        with tf.variable_scope(scope, reuse=reuse):
            inputs_shape = inputs.shape.as_list()
            params_shape = inputs_shape[-1:]
            
            mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
            beta= tf.Variable(tf.zeros(params_shape))
            gamma = tf.Variable(tf.ones(params_shape))
            normalized = (inputs - mean) / ( (variance + epsilon) ** (.5) )
            outputs = gamma * normalized + beta
        return outputs

    def __multihead_attention__(self,
                                queries, 
                                keys, 
                                num_units=None, 
                                num_heads=8, 
                                dropout_rate=0,
                                is_training=True,
                                causality=False,
                                scope="multihead_attention", 
                                reuse=tf.AUTO_REUSE,
                                residual=True):
        '''Applies multihead attention.
        
        Args:
          queries: A 3d tensor with shape of [N, T_q, C_q].
          keys: A 3d tensor with shape of [N, T_k, C_k].
          num_units: A scalar. Attention size.
          dropout_rate: A floating point number.
          is_training: Boolean. Controller of mechanism for dropout.
          causality: Boolean. If true, units that reference the future are masked. 
          num_heads: An int. Number of heads.
          scope: Optional scope for `variable_scope`.
          reuse: Boolean, whether to reuse the weights of a previous layer
            by the same name.
            
        Returns
          A 3d tensor with shape of (N, T_q, C)  
        '''
        with tf.variable_scope(scope, reuse=reuse):
            # Set the fall back option for num_units
            if num_units is None:
                num_units = queries.shape.as_list()[-1]
            # Linear projections
            Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu) # (N, T_q, C)
            K = tf.layers.dense(keys, num_units, activation=tf.nn.relu) # (N, T_k, C)
            V = tf.layers.dense(keys, num_units, activation=tf.nn.relu) # (N, T_k, C)
            score = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))
            outputs = tf.matmul(score, V)
            # Split and concat
            Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0) # (h*N, T_q, C/h) 
            K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0) # (h*N, T_k, C/h) 
            V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0) # (h*N, T_k, C/h) 
            # Multiplication
            outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1])) # (h*N, T_q, T_k)
            # Scale
            outputs = outputs / (float(K_.shape.as_list()[-1]) ** 0.5)
            # Key Masking
            key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1))) # (N, T_k)
            key_masks = tf.tile(key_masks, [num_heads, 1]) # (h*N, T_k)
            key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1]) # (h*N, T_q, T_k)
            paddings = tf.ones_like(outputs)*(-2**32+1)
            outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs) # (h*N, T_q, T_k)
            # Causality = Future blinding
            if causality:
                diag_vals = tf.ones_like(outputs[0, :, :]) # (T_q, T_k)
                tril = tf.contrib.linalg.LinearOperatorTriL(diag_vals).to_dense() # (T_q, T_k)
                masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1]) # (h*N, T_q, T_k)
       
                paddings = tf.ones_like(masks)*(-2**32+1)
                outputs = tf.where(tf.equal(masks, 0), paddings, outputs) # (h*N, T_q, T_k)
            # Activation
            outputs = tf.nn.softmax(outputs) # (h*N, T_q, T_k)
            # Query Masking
            query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1))) # (N, T_q)
            query_masks = tf.tile(query_masks, [num_heads, 1]) # (h*N, T_q)
            query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]]) # (h*N, T_q, T_k)
            outputs *= query_masks # broadcasting. (N, T_q, C)
            # Dropouts
            outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
            # Weighted sum
            outputs = tf.matmul(outputs, V_) # ( h*N, T_q, C/h)
            # Restore shape
            outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2 ) # (N, T_q, C)
            # Residual connection
            if residual:
                outputs += queries
            # Normalize
            outputs = self.__normalize__(outputs) # (N, T_q, C)
        return outputs

    def __feedforward__(self, 
                        inputs, 
                        num_units=[2048, 512],
                        scope="multihead_attention", 
                        reuse=tf.AUTO_REUSE):
        '''Point-wise feed forward net.
        
        Args:
          inputs: A 3d tensor with shape of [N, T, C].
          num_units: A list of two integers.
          scope: Optional scope for `variable_scope`.
          reuse: Boolean, whether to reuse the weights of a previous layer
            by the same name.
            
        Returns:
          A 3d tensor with the same shape and dtype as inputs
        '''
        with tf.variable_scope(scope, reuse=reuse):
            # Inner layer
            params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1,
                      "activation": tf.nn.relu, "use_bias": True}
            outputs = tf.layers.conv1d(**params)
            # Readout layer
            params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1,
                      "activation": None, "use_bias": True}
            outputs = tf.layers.conv1d(**params)
            # Residual connection
            outputs += inputs
            # Normalize
            outputs = self.__normalize__(outputs)
        return outputs

    def attention_is_all_you_need(self, x, hidden_size, num_blocks=4, num_heads=8, activation=tf.nn.relu):
        max_length = x.get_shape()[1]
        self.enc = self.__dropout__(tf.layers.dense(x, hidden_size))
        ## Blocks
        for i in range(num_blocks):
            with tf.variable_scope("num_blocks_{}".format(i)):
                ### Multihead Attention
                self.enc = multihead_attention(queries = self.enc, 
                                                keys = self.enc, 
                                                num_units = hidden_size, 
                                                num_heads = num_heads, 
                                                dropout_rate = self.drop_prob,
                                                is_training = self.is_training,
                                                causality = False)
                ### Feed Forward
                self.enc = feedforward(self.enc, num_units=[4 * hidden_size, hidden_size])
        # piece-wise pooling
        x = self.__piece_pooling__(self.enc, max_length, hidden_size, self.mask)
        x = activation(x)
        return x
