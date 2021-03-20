from bert_serving.client import ConcurrentBertClient


def getEmojiEmbeddings(emojiList,dim=300):
  e2v = gsm.KeyedVectors.load_word2vec_format('emoji2vec.bin', binary=True)
  """ Generates an emoji vector by averaging the emoji representation for each emoji. If no emoji returns an empty list of dimension dim"""
  result = np.zeros(dim)
  if (len(emojiList) == 0):
    return result
  else:
    embs = None
    embs = np.mean([e2v[i] for i in emojiList if i in e2v.vocab], axis=0)
  if np.any(np.isnan(embs)):
    return result
  result[:300] = embs
  return result    

def save_emojifeats(data_dict,s_filename):
    emoji_feats = np.asarray([getEmojiEmbeddings(i) for i in (data_dict['emojis'])])
    with h5py.File(s_filename, "w") as hf:
        hf.create_dataset('feats', data=emoji_feats)

def save_hashfeats(data_dict,s_filename):
    bc = ConcurrentBertClient()
    seg_hashtag =[]
    for hashtag in data_dict['segmented_hashtags']:
        seg_hashtag.append(' '.join(hashtag))
    
    hash_feat = np.zeros((len(seg_hashtag), 768))
    for ind in range(len(seg_hashtag)):
        if seg_hashtag[ind].strip() != '':
            hash_feat[ind] = bc.encode([seg_hashtag[ind].strip()])
    with h5py.File(s_filename, "w") as hf:
        hf.create_dataset('feats', data=hash_feat)