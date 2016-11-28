# Doc2Vec Tensorflow Model

This model describes how to develop a Doc2Vec (Distributed Memory Model) model in tensorflow. **Lines 8-38 describes the model** specification. If you wish to use pre-existing words such as Glove etc., change from `word_embeddings = tf.Variable` to `word_embeddings = tf.placeholder`. This is a modified version of word2vec from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/udacity/5_word2vec.ipynb

**The example does not currently have data to run**. The user needs to provide the ids of the context words, and the id of the document as inputs. (eg. doc ids need to start at 0 and at `len(docs)-1`).

## Notes
1. Doc2Vec needs the document to be fairly large for the document vectors to be meaningful (capture a decent context of the words). This code was not meaningful with tweets. LDA worked better.
2. You can capture more attributes like Author, Genre etc. with these vectors.
