from tensorflow.keras import layers
import tensorflow as tf
import numpy as np

class LSTM:
    def build(self, max_len, vocab_size):
        input1 = layers.Input(shape = (max_len,), dtype = tf.int32)
        input2 = layers.Input(shape = (max_len,), dtype = tf.float32)
        emb = tf.keras.layers.Embedding(input_dim = vocab_size, output_dim = 64, mask_zero = True)(input1)
        emb = tf.multiply(emb, tf.expand_dims(input2, axis = -1))
        outputs = layers.LSTM(units = 64, recurrent_dropout=0.3)(emb)
        outputs = layers.Dense(units = 16, activation = 'relu')(outputs)
        outputs = layers.Dense(units = 1, activation = 'sigmoid')(outputs)[:,0]
        return (input1, input2), outputs

class Transformer:
    def build(self, max_len, vocab_size):
        def positional_encoding(inpt):
            t,d = inpt.get_shape().as_list()[-2:] #static
            n = tf.shape(inpt)[0]
            pos_ind = tf.tile(tf.expand_dims(tf.range(t), 0), [n, 1])
            pos_enc = np.array([
                [pos/np.power(10000, (i-i%2)/d) for i in range(d)]
                for pos in range(t)])
            pos_enc[:, 0::2] = np.sin(pos_enc[:, 0::2])
            pos_enc[:, 1::2] = np.cos(pos_enc[:, 1::2])
            pos_enc = tf.convert_to_tensor(pos_enc, dtype = tf.float32)
            
            return tf.nn.embedding_lookup(pos_enc, pos_ind)
        
        def MultiHeadAttention(query = None, value = None, key = None, 
                               units = None, use_scale = False, num_heads = 4, dropout = 0.0):
            if key is None: key = value
            if units is None: units = value.get_shape().as_list()[-1]
            #Projection
            q_ = layers.Dense(units, activation = 'relu')(query)
            k_ = layers.Dense(units, activation = 'relu')(key)
            v_ = layers.Dense(units, activation = 'relu')(value)
            
            #Split and Concat
            q = tf.concat(tf.split(q_, num_heads, axis = -1), axis = 0)
            k = tf.concat(tf.split(k_, num_heads, axis = -1), axis = 0)
            v = tf.concat(tf.split(v_, num_heads, axis = -1), axis = 0)
            
            #scaled dot production
            scaled_dot_prod = layers.Attention(use_scale = use_scale, dropout = dropout)([q,v,k])
            scaled_dot_prod = tf.concat(tf.split(scaled_dot_prod, num_heads, axis = 0), axis = -1)
            
            #Residual and Normalization
            scaled_dot_prod += query
            
            return layers.LayerNormalization()(scaled_dot_prod)
        
        def FeedForward(raw_inpt, units=[512, 128], dropout = 0.0):
            #FeedForward
            inpt = layers.Dense(units[0], activation = 'relu')(raw_inpt)
            inpt = layers.Dense(units[1])(inpt)
            
            #Dropout, Residual and Normalization
            inpt = layers.Dropout(rate = dropout)(inpt)
            inpt += raw_inpt
            
            return layers.LayerNormalization()(inpt)
        
        input1 = tf.keras.layers.Input(shape = (max_len,), dtype = tf.int32)
        input2 = tf.keras.layers.Input(shape = (max_len,), dtype = tf.float32)
        emb = layers.Embedding(input_dim = vocab_size, output_dim = 64, mask_zero = True)(input1)
        emb *= (8)
        emb *= tf.expand_dims(input2, axis = -1)
        inpt = emb + positional_encoding(emb)
        #stacked multihead attention
        for i in range(3):
            inpt = MultiHeadAttention(query = inpt, 
                                      value = inpt, 
                                      use_scale = True,
                                      num_heads = 4, 
                                      dropout = 0.3)
            inpt = FeedForward(inpt, units = [4*64, 64], dropout = 0.3)
    
        outputs = inpt[:,-1,:]
        outputs = layers.Dense(units = 16, activation = 'relu')(outputs)
        outputs = layers.Dense(units = 1, activation = 'sigmoid')(outputs)[:,0]
        return (input1, input2), outputs
 
class MOE:
    def __init__(self, experts_cnt):
        self.experts_cnt = experts_cnt
        self.gate_LSTM = tf.keras.Sequential([layers.LSTM(units = 32), layers.Dense(units = experts_cnt)])
        self.experts_LSTM = [layers.LSTM(units = 64, recurrent_dropout=0.3) for i in range(experts_cnt)]
    
    def noisy_topk_gate_prob(self, gates):
        k = int(self.experts_cnt // 5)
        values, indices = tf.math.top_k(gates, k)
        sparse_gates = tf.nn.softmax(values)
        bid = tf.tile(tf.expand_dims(tf.range(self.batch_size), axis = -1), [1, k])
        scatter_idx = tf.stack([bid, indices], axis = -1)
        topk_gates = tf.scatter_nd(scatter_idx, sparse_gates, shape = tf.shape(gates))
        return topk_gates
        
    def dispatch(self, emb, gates):
        self.experts_size = tf.reduce_sum(tf.cast(gates > 0, tf.int32), axis = 0)
        nonzero_gates_indices = tf.where(tf.math.greater(gates, 0))
        sort_gates_indices = tf.expand_dims(tf.argsort(nonzero_gates_indices[:,1]), axis = -1)
        gather_idx = tf.gather_nd(nonzero_gates_indices, sort_gates_indices)
        #print('gather_idx_shape', gather_idx.shape)
        nonzero_gates = tf.gather_nd(gates, gather_idx)
        print('nonzero_gates_shape', nonzero_gates.shape)
        self.batch_indices_ = tf.expand_dims(gather_idx[:,0], axis = -1)
        batch_emb = tf.gather_nd(emb, self.batch_indices_)
        return tf.split(batch_emb, self.experts_size, axis = 0), nonzero_gates
    
    def combine(self, expt_outputs, nonzero_gates):
        zeros = tf.zeros([self.batch_size, 64])
        outputs = tf.concat(expt_outputs, axis = 0)
        outputs = tf.multiply(outputs, tf.expand_dims(nonzero_gates, axis = -1))
        return tf.tensor_scatter_nd_add(zeros, self.batch_indices_, outputs)
        
    def build(self, max_len, vocab_size):
        input1 = layers.Input(shape = (max_len,), dtype = tf.int32)
        input2 = layers.Input(shape = (max_len,), dtype = tf.float32)
        self.batch_size = tf.shape(input1)[0]
        emb_type = tf.keras.layers.Embedding(input_dim = vocab_size, output_dim = 64, mask_zero = True)(input1)
        emb = tf.multiply(emb_type, tf.expand_dims(input2, axis = -1))
        # Get gated value for each LSTM
        gates = self.gate_LSTM(emb_type)
        # Add noisy and sparsity on top k experts
        gates = self.noisy_topk_gate_prob(gates)
        # Dispatch samples to each expert based on gate value
        dp_inpt, nonzero_gates = self.dispatch(emb, gates)
        # Forward propagation on each experts_LSTM
        experts_outputs = [expt(inpt) for inpt, expt in zip(dp_inpt, self.experts_LSTM)]
        # weighted sum of the outputs
        outputs = self.combine(experts_outputs, nonzero_gates)
        outputs = layers.Dense(units = 16, activation = 'relu')(outputs)
        outputs = layers.Dense(units = 1, activation = 'sigmoid')(outputs)[:,0]
        return (input1, input2), outputs
        