batch_size = context_window*256
embedding_size = 100 # Dimension of the embedding vector.
softmax_width = embedding_size # +embedding_size2+embedding_size3
num_sampled = 50 # Number of negative examples to sample.
sum_ids = np.repeat(np.arange(batch_size/context_window),context_window)


graph = tf.Graph()

with graph.as_default(), tf.device('/cpu:0'):
    # Input data.
    train_word_dataset = tf.placeholder(tf.int32, shape=[batch_size])
    train_doc_dataset = tf.placeholder(tf.int32, shape=[batch_size/context_window])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size/context_window, 1])

    segment_ids = tf.constant(sum_ids, dtype=tf.int32)

    word_embeddings = tf.Variable(tf.random_uniform([len_words,embedding_size],-1.0,1.0))
    doc_embeddings = tf.Variable(tf.random_uniform([len_docs,embedding_size],-1.0,1.0))

    softmax_weights = tf.Variable(tf.truncated_normal([len_words, softmax_width],
                             stddev=1.0 / np.sqrt(embedding_size)))
    softmax_biases = tf.Variable(tf.zeros([len_words]))

    # Model.
    # Look up embeddings for inputs.
    embed_words = tf.segment_sum(tf.nn.embedding_lookup(word_embeddings, train_word_dataset),segment_ids)
    embed_docs = tf.nn.embedding_lookup(doc_embeddings, train_doc_dataset)
    embed = embed_words+embed_docs#+embed_hash+embed_users

    # Compute the softmax loss, using a sample of the negative labels each time.
    loss = tf.reduce_mean(tf.nn.nce_loss(softmax_weights, softmax_biases, embed,
                                   train_labels, num_sampled, len_words))

    # Optimizer.
    optimizer = tf.train.AdagradOptimizer(0.5).minimize(loss)

############################
# Chunk the data to be passed into the tensorflow Model
###########################
data_idx = 0
def generate_batch(batch_size, context_window, x, document, target):
    global data_idx

    batch_labels = collections.deque(maxlen=batch_size)
    batch_doc_data = collections.deque(maxlen=batch_size)
    batch_word_data = collections.deque(maxlen=batch_size*context_window)
    for _ in range(batch_size):
        batch_doc_data.append(document[data_idx])
        batch_labels.append(target[data_idx])

#         idx = np.where(hashtag_doc==document[data_idx])
        for i in range(context_window):
            batch_word_data.append(x[context_window*data_idx+i])
        data_idx = (data_idx + 1) % len(data)

    return batch_labels, batch_word_data, batch_doc_data

num_steps = 200001
step_delta = int(num_steps/20)

with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    print('Initialized')
    average_loss = 0
    for step in range(num_steps):
        batch_labels, batch_word_data, batch_doc_data\
        = generate_batch(int(batch_size/context_window), context_window,
                         x, document, target)
        feed_dict = {train_word_dataset : batch_word_data,
                     train_doc_dataset : batch_doc_data,
                     train_labels : batch_labels}
        _, l = session.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += l
        if step % step_delta == 0:
            if step > 0:
                average_loss = average_loss / step_delta
            # The average loss is an estimate of the loss over the last 2000 batches.
            print('Average loss at step %d: %f' % (step, average_loss))
            average_loss = 0

    # Get the weights to save for later
    final_doc_embeddings = doc_embeddings.eval()
    final_word_embeddings = word_embeddings.eval()
    final_word_embeddings_out = softmax_weights.eval()
