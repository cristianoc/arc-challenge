import tensorflow as tf

# Disable all GPUs
tf.config.set_visible_devices([], "GPU")

# Verify that the GPU is disabled
print("Available GPUs:", tf.config.list_physical_devices("GPU"))

# Your TensorFlow code here


# import tensorflow as tf


# # Create a simple computation to verify GPU usage
# a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
# b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
# c = tf.matmul(a, b)

# print("\n\n")
# # Check if TensorFlow is using the GPU
# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices("GPU")))
# print("Result of matrix multiplication:\n", c)
