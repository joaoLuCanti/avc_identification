import tensorflow as tf

tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
print("Rodando na tpu: ", tpu.cluster_spec().as_dict())