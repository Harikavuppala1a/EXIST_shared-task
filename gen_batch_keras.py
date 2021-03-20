import numpy as np
import keras

class TrainGenerator(keras.utils.Sequence):
    def __init__(self, list_IDs, labels, word_feats, sent_feats, data_dict, batch_size,use_emojis,emoji_array,use_hashtags, hashtag_array):
        self.word_feats = word_feats
        self.sent_feats = sent_feats
        self.data_dict = data_dict

        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.use_emojis = use_emojis
        self. emoji_array = emoji_array
        self.use_hashtags= use_hashtags
        self. hashtag_array = hashtag_array
        # print (len(self.list_IDs))

    def __len__(self):
        # print (int(np.ceil(len(self.list_IDs) / self.batch_size)))
        return int(np.ceil(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        list_IDs_temp = self.list_IDs[index*self.batch_size:min(len(self.list_IDs),(index+1)*self.batch_size)]
        # print ("listtttttttt temps")
        # print (len(list_IDs_temp))
        X, y = self.__data_generation(list_IDs_temp)
        return X, y

    def on_epoch_end(self):
        # print ("epoch done")
        np.random.shuffle(self.list_IDs)

    def __data_generation(self, list_IDs_temp):
        # print ("i am generating data")
        X_inputs = []
        for word_feat in self.word_feats:
            X_inputs.append(np.empty((len(list_IDs_temp), *word_feat['dim_shape'])))

        for i, ID in enumerate(list_IDs_temp):
            input_ind = 0       
            for word_feat in self.word_feats:
                if 'func' in word_feat:
                    X_inputs[input_ind][i,] = word_feat['func'](ID, word_feat, self.data_dict, word_feat['filepath'] + str(ID) + '.npy', word_feat['emb_size'])
                else:
                    X_inputs[input_ind][i,] = np.load(word_feat['filepath'] + str(ID) + '.npy')
                input_ind += 1

        for sent_feat in self.sent_feats:
            X_inputs.append(sent_feat['feats'][list_IDs_temp])

        if self.use_emojis:
            X_inputs.append(self.emoji_array[list_IDs_temp])                  # X_inputs.append(self.conf_scores_array[list_IDs_temp])

        if self.use_hashtags:
            X_inputs.append(self.hashtag_array[list_IDs_temp])
            

        return X_inputs, self.labels[list_IDs_temp]

class TestGenerator(keras.utils.Sequence):
    def __init__(self, list_IDs, word_feats, sent_feats, data_dict, batch_size, use_emojis,emoji_array,use_hashtags, hashtag_array):
        self.word_feats = word_feats
        self.sent_feats = sent_feats
        self.data_dict = data_dict

        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.use_emojis = use_emojis
        self. emoji_array = emoji_array
        self.use_hashtags= use_hashtags
        self. hashtag_array = hashtag_array

    def __len__(self):
        return int(np.ceil(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        list_IDs_temp = self.list_IDs[index*self.batch_size:min(len(self.list_IDs),(index+1)*self.batch_size)]

        X = self.__data_generation(list_IDs_temp)
        return X

    def __data_generation(self, list_IDs_temp):
        X_inputs = []
        for word_feat in self.word_feats:
            X_inputs.append(np.empty((len(list_IDs_temp), *word_feat['dim_shape'])))

        for i, ID in enumerate(list_IDs_temp):
            input_ind = 0       
            for word_feat in self.word_feats:
                if 'func' in word_feat:
                    X_inputs[input_ind][i,] = word_feat['func'](ID, word_feat, self.data_dict, word_feat['filepath'] + str(ID) + '.npy', word_feat['emb_size'])
                else:
                    X_inputs[input_ind][i,] = np.load(word_feat['filepath'] + str(ID) + '.npy')
                input_ind += 1

        for sent_feat in self.sent_feats:
            X_inputs.append(sent_feat['feats'][list_IDs_temp])

        if self.use_emojis:
            X_inputs.append(self.emoji_array[list_IDs_temp])

        if self.use_hashtags:
            X_inputs.append(self.hashtag_array[list_IDs_temp])

        return X_inputs