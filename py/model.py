import tensorflow as tf

class Model(object):
  def __init__(self, observations_dims, observations_rows, observations_cols, 
      actions_dims, hidden_dims, lr, reg_factor, loss):
    # Define inputs
    self.observations = tf.placeholder(
      tf.int32, shape=(None, observations_dims))
    self.actions = tf.placeholder(tf.int32, shape=(None))
    self.rewards = tf.placeholder(tf.float32, shape=(None))
    self.temperature = tf.placeholder_with_default(1.0, [])

    batch_size = tf.shape(self.observations)[0]

    # Lookup observations' embeddings
    obs_emb_w = tf.get_variable(
      "obs_embedding", [observations_rows, observations_cols])
    obs_emb = tf.gather(obs_emb_w, tf.reshape(self.observations, [-1]))
    obs_emb = tf.reshape(
      obs_emb, tf.pack([batch_size, observations_dims * observations_cols]))
    obs_emb.set_shape([None, observations_dims * observations_cols])

    # Nonlinearities
    last_hidden = obs_emb
    for i, dim in enumerate(hidden_dims):
      last_hidden = tf.contrib.layers.fully_connected(
        last_hidden, dim, scope="hidden_%d" % i)

    # Calculate probabilities for all actions for output
    logits = tf.contrib.layers.fully_connected(
      last_hidden, actions_dims, activation_fn=None,
      scope="output")

    # Calculate log-likelihoods for given action
    if loss == "softmax":
      self.outputs = tf.nn.softmax(logits / self.temperature)
      lls = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits, self.actions)
      loss_term = tf.reduce_mean(self.rewards * lls)
    elif loss == "l2":
      self.outputs = logits
      offsets = tf.range(batch_size) * actions_dims
      vals = tf.gather(tf.reshape(logits, [-1]), self.actions + offsets)
      loss_term = tf.nn.l2_loss(self.rewards - vals)
    else:
      print "Unknown loss:", loss

    # Calculate weight regularization term
    reg_term = tf.add_n(
      [tf.nn.l2_loss(var) for var in tf.trainable_variables()])

    # Maximize stochastic expectation of rewards
    self.loss = loss_term + reg_term * reg_factor
    self.global_step = tf.Variable(0, name='global_step', trainable=False)
    self.optimize = tf.train.AdagradOptimizer(lr).minimize(
      self.loss, global_step=self.global_step)
