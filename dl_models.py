import tensorflow as tf
from keras import backend as K
from keras.layers import TimeDistributed, Embedding, Dense, Input, Flatten, Conv1D, GlobalMaxPooling1D, Dropout, LSTM, GRU, Bidirectional, concatenate, Lambda
from keras.models import Model
from keras import optimizers
from keras.engine.topology import Layer
from keras import initializers
import numpy as np
import tensorflow_hub as hub
from keras.utils import multi_gpu_model

def flat_embed(enc_algo, word_emb_seq, word_cnt_post, word_emb_len, num_cnn_filters, max_pool_k_val, kernel_sizes, rnn_dim, att_dim, rnn_type, att_outputs):
    if enc_algo == "rnn":
        if att_dim > 0:
            rnn_mod, att_mod = rnn_sen_embed(word_cnt_post, word_emb_len, rnn_dim, att_dim, rnn_type)
            att_outputs.append(att_mod(word_emb_seq))
        else:
            rnn_mod = rnn_sen_embed(word_cnt_post, word_emb_len, rnn_dim, att_dim, rnn_type)
        # rnn_mod.summary()
        return rnn_mod(word_emb_seq), att_outputs
    elif enc_algo == "cnn":
        cnn_mod = cnn_sen_embed(word_cnt_post, word_emb_len, num_cnn_filters, max_pool_k_val, kernel_sizes)
        # cnn_mod.summary()
        return cnn_mod(word_emb_seq), att_outputs
    elif enc_algo == "c_rnn":
        if att_dim > 0:
            c_rnn_mod, att_mod = c_rnn_sen_embed(word_cnt_post, word_emb_len, rnn_dim, att_dim, rnn_type, num_cnn_filters, kernel_sizes)
            att_outputs.append(att_mod(word_emb_seq))
        else:
            c_rnn_mod = c_rnn_sen_embed(word_cnt_post, word_emb_len, rnn_dim, att_dim, rnn_type, num_cnn_filters, kernel_sizes)
        # c_rnn_mod.summary()
        return c_rnn_mod(word_emb_seq), att_outputs
    elif enc_algo == "comb_cnn_rnn":
        if att_dim > 0:
            rnn_mod, att_mod = rnn_sen_embed(word_cnt_post, word_emb_len, rnn_dim, att_dim, rnn_type)
            rnn_emb_output = rnn_mod(word_emb_seq)
            att_outputs.append(att_mod(word_emb_seq))
        else:
            rnn_emb_output = rnn_sen_embed(word_cnt_post, word_emb_len, rnn_dim, att_dim, rnn_type)(word_emb_seq)
        cnn_emb_output = cnn_sen_embed(word_cnt_post, word_emb_len, num_cnn_filters, max_pool_k_val, kernel_sizes)(word_emb_seq)

        return concatenate([cnn_emb_output, rnn_emb_output]), att_outputs

def rnn_sen_embed(word_cnt_sent, word_emb_len, rnn_dim, att_dim, rnn_type):
    w_emb_input_seq = Input(shape=(word_cnt_sent, word_emb_len), name='emb_input')
    return rnn_generic_embed(w_emb_input_seq, w_emb_input_seq, rnn_dim, att_dim, rnn_type)

def c_rnn_sen_embed(word_cnt_sent, word_emb_len, rnn_dim, att_dim, rnn_type, num_cnn_filters, kernel_sizes):
    w_emb_input_seq = Input(shape=(word_cnt_sent, word_emb_len), name='emb_input')
    conv_l_list = []
    for k in kernel_sizes:
        conv_t = Conv1D(num_cnn_filters, k, padding='same', activation='relu')(w_emb_input_seq)
        conv_l_list.append(conv_t)
    conc_mat = concatenate(conv_l_list)
    return rnn_generic_embed(w_emb_input_seq, conc_mat, rnn_dim, att_dim, rnn_type)

def rnn_generic_embed(w_emb_input_seq, seq_for_rnn, rnn_dim, att_dim, rnn_type):
    if rnn_type == 'lstm':
        blstm_l = Bidirectional(LSTM(rnn_dim, return_sequences=(att_dim > 0)))(seq_for_rnn)
    elif rnn_type == 'gru':
        blstm_l = Bidirectional(GRU(rnn_dim, return_sequences=(att_dim > 0)))(seq_for_rnn)
    if att_dim > 0:
        blstm_l, att_w = attLayer_hier(att_dim)(blstm_l)
        return Model(w_emb_input_seq, blstm_l), Model(w_emb_input_seq, att_w)
    else:
        return Model(w_emb_input_seq, blstm_l)

def cnn_sen_embed(word_cnt_sent, word_emb_len, num_cnn_filters, max_pool_k_val, kernel_sizes):
    w_emb_input_seq = Input(shape=(word_cnt_sent, word_emb_len), name='emb_input')
    conv_l_list = []
    for k in kernel_sizes:
        conv_t = Conv1D(num_cnn_filters, k, padding='same', activation='relu')(w_emb_input_seq)
        if max_pool_k_val == 1:
            pool_t = GlobalMaxPooling1D()(conv_t)
        else:
            pool_t = kmax_pooling(max_pool_k_val)(conv_t)
        conv_l_list.append(pool_t)
    feat_vec = concatenate(conv_l_list)
    return Model(w_emb_input_seq, feat_vec)

def post_embed(sen_emb, rnn_dim, att_dim, rnn_type, stack_rnn_flag, att_outputs):
    if rnn_type == 'lstm':
        blstm_l = Bidirectional(LSTM(rnn_dim, return_sequences=(att_dim > 0)))(sen_emb)
        if stack_rnn_flag:
            blstm_l = Bidirectional(LSTM(rnn_dim, return_sequences=(att_dim > 0)))(blstm_l)
    elif rnn_type == 'gru':
        blstm_l = Bidirectional(GRU(rnn_dim, return_sequences=(att_dim > 0)))(sen_emb)
        if stack_rnn_flag:
            blstm_l = Bidirectional(GRU(rnn_dim, return_sequences=(att_dim > 0)))(blstm_l)
    if att_dim > 0:
        blstm_l, att_w = attLayer_hier(att_dim)(blstm_l)
        att_outputs.append(att_w)
    return blstm_l, att_outputs


def add_word_emb_p_flat(model_inputs, word_emb_input, word_f_word_emb, word_f_emb_size, enc_algo, m_id, p_dict):
        model_inputs.append(word_emb_input)
        if m_id in p_dict:
            p_dict[m_id]["comb_feature_list"].append(word_f_word_emb)
            p_dict[m_id]["word_emb_len"] += word_f_emb_size
            p_dict[m_id]["enc_algo"] = enc_algo
        else:
            p_dict[m_id] = {}
            p_dict[m_id]["comb_feature_list"] = [word_f_word_emb]
            p_dict[m_id]["word_emb_len"] = word_f_emb_size 
            p_dict[m_id]["enc_algo"] = enc_algo

class BertLayer(Layer):
    
    '''BertLayer which support next output_representation param:
    
    pooled_output: the first CLS token after adding projection layer () with shape [batch_size, 768]. 
    sequence_output: all tokens output with shape [batch_size, max_length, 768].
    mean_pooling: mean pooling of all tokens output [batch_size, max_length, 768].
    
    
    You can simple fine-tune last n layers in BERT with n_fine_tune_layers parameter. For view trainable parameters call model.trainable_weights after creating model.
    
    '''
    
    def __init__(self, n_fine_tune_layers, tf_hub_path, output_representation,trainable,**kwargs):
        
        self.n_fine_tune_layers = n_fine_tune_layers
        self.is_trainble = trainable
        self.output_size = 768
        self.tf_hub = tf_hub_path
        self.output_representation = output_representation
        
        super(BertLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        self.bert = hub.Module(
            self.tf_hub,
            trainable=self.is_trainble,
            name="{}_module".format(self.name)
        )
        
        
        variables = list(self.bert.variable_map.values())
        print (len(variables))
        # for var in variables:
        #     print(var)
        # print("------------------")
        # for var in self.bert.variables:
        #     print(var)
        # exit()
        if self.is_trainble:
            # 1 first remove unused layers
            trainable_vars = [var for var in variables if not "/cls/" in var.name]
            
            print (len(trainable_vars))
            if self.output_representation == "sequence_output" or self.output_representation == "mean_pooling":
                # 1 first remove unused pooled layers
                trainable_vars = [var for var in trainable_vars if not "/pooler/" in var.name]
                
            # Select how many layers to fine tune
            trainable_vars = trainable_vars[-self.n_fine_tune_layers :]
            print (len(trainable_vars))
            # Add to trainable weights
            for var in trainable_vars:
                self._trainable_weights.append(var)

            # Add non-trainable weights
            for var in self.bert.variables:
                if var not in self._trainable_weights:
                    self._non_trainable_weights.append(var)
                
        else:
             for var in variables:
                self._non_trainable_weights.append(var)
                

        super(BertLayer, self).build(input_shape)

    def call(self, inputs):
        inputs = [K.cast(x, dtype="int32") for x in inputs]
        input_ids, input_mask, segment_ids = inputs
        bert_inputs = dict(
            input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids
        )
        result = self.bert(inputs=bert_inputs, signature="tokens", as_dict=True)
        
        if self.output_representation == "pooled_output":
            pooled = result["pooled_output"]
            
        elif self.output_representation == "mean_pooling":
            result_tmp = result["sequence_output"]
        
            mul_mask = lambda x, m: x * tf.expand_dims(m, axis=-1)
            masked_reduce_mean = lambda x, m: tf.reduce_sum(mul_mask(x, m), axis=1) / (
                    tf.reduce_sum(m, axis=1, keepdims=True) + 1e-10)
            input_mask = tf.cast(input_mask, tf.float32)
            pooled = masked_reduce_mean(result_tmp, input_mask)
            
        elif self.output_representation == "sequence_output":
            
            pooled = result["sequence_output"]
       
        return pooled
        
        
    def compute_output_shape(self, input_shape):
        if self.output_representation == "sequence_output":
            return (input_shape[0][0], input_shape[0][1], self.output_size)
        else:
            return (input_shape[0][0], self.output_size)

def gen_tune_bert_model(sen_f_input,comp_dim,rnn_dim,att_dim,n_fine_tune_layers,tf_hub_path, output_representation, bert_trainable):
      
    b_input = Input(shape=(comp_dim,))
    ind_len = int(comp_dim/3)
    in_id = Lambda(lambda x: x[:,0:ind_len], output_shape=(ind_len,))(sen_f_input)
    in_mask = Lambda(lambda x: x[:,ind_len:2*ind_len], output_shape=(ind_len,))(sen_f_input)
    in_segment = Lambda(lambda x: x[:,2*ind_len:3*ind_len], output_shape=(ind_len,))(sen_f_input)
    bert_inputs = [in_id, in_mask, in_segment]
    bert_output = BertLayer(n_fine_tune_layers,tf_hub_path, output_representation, bert_trainable)(bert_inputs)

    if output_representation == "sequence_output":

        bert_output = Bidirectional(LSTM(rnn_dim, return_sequences=(att_dim > 0)))(bert_output)
        if att_dim > 0:
            bert_output, att_w = attLayer_hier(att_dim)(bert_output)
    # print (blstm_l.shape)
    # m = Model(sen_f_input,bert_output)
    # m.summary()
    print (bert_output.shape)
    return bert_output

def flat_fuse(word_cnt_post, rnn_dim, att_dim, word_feats, sen_enc_feats, dropO1, dropO2, nonlin, out_vec_size, rnn_type, stack_rnn_flag, num_cnn_filters, max_pool_k_val, kernel_sizes,n_fine_tune_layers,tf_hub_path, output_representation, bert_trainable,use_emotions, emoji_array,use_hashtags, hashtag_array,use_empath, empath_array,use_perspective,perspective_array,use_hurtlex,hurtlex_array):
    p_dict = {}
    model_inputs = []
    att_outputs = []

    for word_feat in word_feats:
        if 'embed_mat' in word_feat:
            word_f_input, word_f_word_emb_raw = tunable_embed_apply(word_cnt_post, len(word_feat['embed_mat']), word_feat['embed_mat'], word_feat['emb'])
            word_f_word_emb = Dropout(dropO1)(word_f_word_emb_raw)
            add_word_emb_p_flat(model_inputs, word_f_input, word_f_word_emb, word_feat['embed_mat'].shape[-1], word_feat['s_enc'], word_feat['m_id'], p_dict)
        else:
            word_f_input = Input(shape=(word_cnt_post, word_feat['dim_shape'][-1]), name=word_feat['emb'])
            word_f_word_emb = Dropout(dropO1)(word_f_input)
            add_word_emb_p_flat(model_inputs, word_f_input, word_f_word_emb, word_feat['dim_shape'][-1], word_feat['s_enc'], word_feat['m_id'], p_dict)

    post_vec_list = []    
    for my_dict in p_dict.values():
        my_dict["word_emb"] = concatenate(my_dict["comb_feature_list"]) if len(my_dict["comb_feature_list"]) > 1 else my_dict["comb_feature_list"][0]
        flat_emb, att_outputs = flat_embed(my_dict["enc_algo"], my_dict["word_emb"], word_cnt_post, my_dict["word_emb_len"], num_cnn_filters, max_pool_k_val, kernel_sizes, rnn_dim, att_dim, rnn_type, att_outputs)
        post_vec_list.append(flat_emb)
    print (len(post_vec_list))

    for sen_enc_feat in sen_enc_feats:
        sen_f_input = Input(shape=(sen_enc_feat['feats'].shape[-1],), name=sen_enc_feat['emb'])
        model_inputs.append(sen_f_input) 
        print (sen_f_input.shape)
        if sen_enc_feat['emb'].startswith('trainable'):
            sen_f_dr1 = gen_tune_bert_model(sen_f_input, sen_enc_feat['feats'].shape[-1],rnn_dim,att_dim,n_fine_tune_layers,tf_hub_path, output_representation, bert_trainable)
            # print ( tune_bert_model.shape)
            # sen_f_dr1 = tune_bert_model(sen_f_input)
            # print (sen_f_dr1.shape)
            # sen_f_dr = tune_bert_model(sen_f_input)
            # print (tune_bert_model.shape)
            # sen_f_dr1= Dropout(dropO1)(tune_bert_model)
            print (sen_f_dr1)
        else:       
            sen_f_dr1 = Dropout(dropO1)(sen_f_input)
        post_vec_list.append(sen_f_dr1)

    if use_emotions:
        emo_input = Input(shape=(emoji_array.shape[-1],))
        model_inputs.append(emo_input)
        # sen_f_dr1 = Dropout(dropO1)(emo_input)
        post_vec_list.append(emo_input)
    if use_hashtags:
        hash_input = Input(shape=(hashtag_array.shape[-1],))
        model_inputs.append(hash_input)
        # sen_f_dr1 = Dropout(dropO1)(hash_input)
        post_vec_list.append(hash_input)
    if use_empath:
        empath_input = Input(shape=(empath_array.shape[-1],))
        model_inputs.append(empath_input)
        # sen_f_dr1 = Dropout(dropO1)(empath_input)
        post_vec_list.append(empath_input)
    if use_perspective:
        perspective_input = Input(shape=(perspective_array.shape[-1],))
        model_inputs.append(perspective_input)
        # sen_f_dr1 = Dropout(dropO1)(perspective_input)
        post_vec_list.append(perspective_input)
    if use_hurtlex:
        hurtlex_input = Input(shape=(hurtlex_array.shape[-1],))
        model_inputs.append(hurtlex_input)
        post_vec_list.append(hurtlex_input)

    # post_vec_list
    post_vec = concatenate(post_vec_list) if len(post_vec_list) > 1 else post_vec_list[0]
    print(post_vec.shape)

    att_mod = Model(model_inputs, att_outputs) if att_outputs else None
    model,out_vec = apply_dense(model_inputs, dropO2, post_vec, nonlin, out_vec_size)
    return model,out_vec, att_mod,model_inputs

def apply_dense(input_seq, dropO2, post_vec, nonlin, out_vec_size):
    dr2_l = Dropout(dropO2)(post_vec)
    out_vec = Dense(out_vec_size, activation=nonlin)(dr2_l)
    print (out_vec.shape)
    return Model(input_seq, out_vec), out_vec

def tunable_embed_apply(word_cnt_post, vocab_size, embed_mat, word_feat_name):
    input_seq = Input(shape=(word_cnt_post,), name=word_feat_name+'_t')
    embed_layer = Embedding(vocab_size, embed_mat.shape[1], embeddings_initializer=initializers.Constant(embed_mat), input_length=word_cnt_post, name=word_feat_name)
    embed_layer.trainable = True
    embed_l = embed_layer(input_seq)
    return input_seq, embed_l

# adapted from https://github.com/richliao/textClassifier
class attLayer_hier(Layer):
    def __init__(self, attention_dim, **kwargs):
        self.init = initializers.get('glorot_uniform')
        # self.init = initializers.get('normal')
        # self.supports_masking = True
        self.attention_dim = attention_dim
        super(attLayer_hier, self).__init__(**kwargs)

    def build(self, input_shape):
        # assert len(input_shape) == 3
        self.W = self.add_weight(name = 'W', shape = (input_shape[-1], self.attention_dim), initializer=self.init, trainable=True)
        self.b = self.add_weight(name = 'b', shape = (self.attention_dim, ), initializer=self.init, trainable=True)
        self.u = self.add_weight(name = 'u', shape = (self.attention_dim, 1), initializer=self.init, trainable=True)
        super(attLayer_hier, self).build(input_shape)

    # def compute_mask(self, inputs, mask=None):
    #     return mask

    def call(self, x, mask=None):
        uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))
        ait = K.dot(uit, self.u)
        ait = K.squeeze(ait, -1)
        ait = K.exp(ait)

        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            ait *= K.cast(mask, K.floatx())

        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        exp_ait = K.expand_dims(ait)
        weighted_input = x * exp_ait
        output = K.sum(weighted_input, axis=1)
        # print(K.int_shape(ait))

        return [output, ait]

    def compute_output_shape(self, input_shape):
        return [(input_shape[0], input_shape[-1]), (input_shape[0], input_shape[1])]

    def get_config(self):
        config = {'attention_dim': self.attention_dim}
        base_config = super(attLayer_hier, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class kmax_pooling(Layer):
    def __init__(self, k_val, **kwargs):
        self.k_val = k_val
        super(kmax_pooling, self).__init__(**kwargs)

    def call(self, inputs):
        
        # swap last two dimensions since top_k will be applied along the last dimension
        shifted_input = tf.transpose(inputs, [0, 2, 1])
        
        # extract top_k, returns two tensors [values, indices]
        top_k_var = tf.nn.top_k(shifted_input, k=self.k_val, sorted=True, name=None)[0]
        
        # return flattened output
        return tf.reshape(top_k_var, [tf.shape(top_k_var)[0], -1])

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1]* self.k_val)

    def get_config(self):
        config = {'k_val': self.k_val}
        base_config = super(kmax_pooling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def lp_categ_loss(weights):
    def lp_categ_of(y_true, y_pred):
        # return K.squeeze(K.dot(y_true, K.expand_dims(K.variable(weights))), -1)
        return K.sum(weights*y_true,axis=1)*K.categorical_crossentropy(y_true, y_pred)
    return lp_categ_of

def br_binary_loss(weights):
    def br_binary_of(y_true, y_pred):
        return ((weights[0]*(1-y_true))+(weights[1]*y_true))*K.binary_crossentropy(y_true, y_pred)
        # return weights[y_true]*K.binary_crossentropy(y_true, y_pred)
    return br_binary_of


def get_model(m_type, word_cnt_post, sent_cnt, word_cnt_sent, word_feats, sen_enc_feats, learn_rate, dropO1, dropO2, num_cnn_filters, rnn_type, loss_func, nonlin, out_vec_size, rnn_dim, att_dim, max_pool_k_val, stack_rnn_flag, kernel_sizes,n_fine_tune_layers,tf_hub_path, output_representation, bert_trainable, use_emotions, emoji_array,use_hashtags, hashtag_array,use_empath, empath_array,use_perspective,perspective_array,use_hurtlex,hurtlex_array):
    model, out_vec, att_mod, model_inputs= flat_fuse(word_cnt_post, rnn_dim, att_dim, word_feats, sen_enc_feats, dropO1, dropO2, nonlin, out_vec_size, rnn_type, stack_rnn_flag, num_cnn_filters, max_pool_k_val, kernel_sizes,n_fine_tune_layers,tf_hub_path, output_representation, bert_trainable, use_emotions, emoji_array,use_hashtags, hashtag_array,use_empath, empath_array,use_perspective,perspective_array,use_hurtlex,hurtlex_array)
    # print (model_old.summary())
    print (model.summary())
    adam = optimizers.Adam(lr=learn_rate)
    model.compile(loss=loss_func, optimizer=adam)
    return model, att_mod

# import numpy as np
# word_feats = [{'emb': 'elmo', 's_enc': 'rnn', 'm_id': '11', 'dim_shape': (100,3000)}]#, {'emb': 'glove', 's_enc': 'rnn', 'm_id': '21','dim_shape': [100,300]}
# sen_enc_feats = [{'emb': 'use', 'm_id': '1', 'feats': np.zeros((100, 768))},{'emb': 'bert', 'm_id': '1', 'feats': np.zeros((100, 1024))}]
# m, a = get_model('hier_fuse', 200, 20, 100, word_feats, sen_enc_feats, 0.001, 0.1, 0.1, 80, 'lstm', 'binary_crossentropy', 'sigmoid', 14, 111, 333, 9, False, [2,3,4])
# m.summary()
# a.summary()