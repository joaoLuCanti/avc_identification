import tensorflow as tf

tpu = tf.distribute.cluster_resolver.TPUClusterResolver("local")
print("Rodando na tpu: ", tpu.cluster_spec().as_dict())