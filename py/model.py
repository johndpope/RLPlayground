import tensorflow as tf


def DenseLayers(last, dims):
  for i, dim in enumerate(dims):
    last = tf.contrib.layers.fully_connected(
      last, dim, scope="hidden_%d" % i)
  return last


def ResidualLayers(last, dims):
  ll = last
  ll_dim = last.get_shape().as_list()[-1]
  for i, dim in enumerate(dims):
    last = tf.contrib.layers.fully_connected(
      tf.nn.relu(last), dim, activation_fn=None, scope="hidden_%d" % i)
    if i % 2 == 1:
      if dim != ll_dim:
        ll = tf.contrib.layers.fully_connected(
          ll, dim, activation_fn=None, scope="proj_%d" % i)
      last += ll 
      ll = last
      ll_dim = dim
  last = tf.nn.relu(last)
  return last


class Model(object):
  def __init__(self, observations_dims, observations_rows, observations_cols, 
      actions_dims, hidden_dims, lr, reg_factor, loss, use_residual=False,
      b_factor=0.9):
    # Define inputs
    self.observations = tf.placeholder(
      tf.int32, shape=(None, observations_dims))
    self.actions = tf.placeholder(tf.int32, shape=(None))
    self.targets = tf.placeholder(tf.float32, shape=(None))

    batch_size = tf.shape(self.observations)[0]

    # Lookup observations' embeddings
    obs_emb_w = tf.get_variable(
      "obs_embedding", [observations_rows, observations_cols])
    obs_emb = tf.gather(obs_emb_w, tf.reshape(self.observations, [-1]))
    obs_emb = tf.reshape(
      obs_emb, tf.stack([batch_size, observations_dims * observations_cols]))
    obs_emb.set_shape([None, observations_dims * observations_cols])

    # Nonlinearities
    if use_residual:
      hidden = ResidualLayers(obs_emb, hidden_dims)
    else:
      hidden = DenseLayers(obs_emb, hidden_dims)

    # Calculate probabilities for all actions for output
    logits = tf.contrib.layers.fully_connected(
      hidden, actions_dims, activation_fn=None,
      scope="output")

    # Calculate log-likelihoods for given action
    if loss == "softmax":
      self.outputs = tf.nn.softmax(logits)
      lls = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=self.actions, logits=logits)
      baseline = tf.Variable(0., name="baseline")
      tf.summary.scalar('loss/baseline', baseline)
      tf.summary.scalar('loss/rewards', tf.reduce_mean(self.targets))
      b_term = b_factor * tf.nn.l2_loss(self.targets - baseline)
      adjusted_rewards = tf.stop_gradient(self.targets - baseline)
      loss_term = tf.reduce_mean(adjusted_rewards * lls) + b_term      
    elif loss == "l2":
      self.outputs = logits
      offsets = tf.range(batch_size) * actions_dims
      vals = tf.gather(tf.reshape(logits, [-1]), self.actions + offsets)
      loss_term = tf.nn.l2_loss(self.targets - vals)
      tf.summary.scalar('loss/l2_loss', loss_term)
    else:
      print "Unknown loss:", loss

    # Calculate weight regularization term
    reg_term = tf.add_n(
      [tf.nn.l2_loss(var) for var in tf.trainable_variables()])

    # Maximize stochastic expectation of targets
    self.loss = loss_term + reg_term * reg_factor
    tf.summary.scalar('loss/loss', self.loss)

    self.global_step = tf.Variable(0, name='global_step', trainable=False)
    self.optimize = tf.train.AdagradOptimizer(lr).minimize(
      self.loss, global_step=self.global_step)
    
    self.summaries = tf.summary.merge_all()
