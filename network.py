import tensorflow as tf
import tensorflow_datasets as tfds

ds = tfds.load('mnist', split='train', shuffle_files=True)
ds = ds.take(1)

print(next(iter(ds)))


def network():
    print(ds)
