import tensorflow as tf

# Verifique se o TensorFlow reconhece as GPUs
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("GPUs disponíveis:")
    for gpu in gpus:
        print(gpu)
else:
    print("Nenhuma GPU disponível.")

from tensorflow.python.client import device_lib

# Listar todos os dispositivos físicos disponíveis
print("Dispositivos físicos disponíveis:")
print(device_lib.list_local_devices())