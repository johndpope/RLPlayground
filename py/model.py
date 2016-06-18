import tensorflow as tf

class Model(object):
  def __init__(self, input_dim, embedding_rows, embedding_cols, output_dim, 
      hidden_dims, lr, reg_factor):
    # Define inputs
    self.observations = tf.placeholder(tf.int32, shape=(None, input_dim)) 
    self.actions = tf.placeholder(tf.int32, shape=(None))
    self.rewards = tf.placeholder(tf.float32, shape=(None))
    self.temperature = tf.placeholder_with_default(1.0, [])

    batch_size = tf.shape(self.observations)[0]

    # Calculate feature embeddings
    embedding_w = tf.get_variable(
      "embedding", [embedding_rows, embedding_cols])
    embeddings = tf.gather(embedding_w, tf.reshape(self.observations, [-1]))
    embeddings = tf.reshape(
      embeddings, tf.pack([batch_size, input_dim * embedding_cols]))
    embeddings.set_shape([None, input_dim * embedding_cols])

    # Nonlinearities
    last_hidden = embeddings
    for i, dim in enumerate(hidden_dims):
      last_hidden = tf.contrib.layers.fully_connected(
        last_hidden, dim, scope="hidden_%d" % i)

    # Calculate probabilities for all actions for output
    logits = tf.contrib.layers.fully_connected(
      last_hidden, output_dim, activation_fn=None, 
      scope="output")
    self.outputs = tf.nn.softmax(logits / self.temperature)

    # Calculate log-likelihoods for given action 
    lls = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits, self.actions)
    
    # Calculate weight regularization term
    reg_term = tf.add_n(
      [tf.nn.l2_loss(var) for var in tf.trainable_variables()])
    
    # Maximize stochastic expectation of rewards   
    self.loss = tf.reduce_mean(self.rewards * lls) + reg_term * reg_factor
    self.global_step = tf.Variable(0, name='global_step', trainable=False)
    self.optimize = tf.train.AdagradOptimizer(lr).minimize(
      self.loss, global_step=self.global_step)
