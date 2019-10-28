import runway
import numpy as np
from tensorflow.python import keras


@runway.setup
def setup():
    model = keras.models.load_model('models/generator.h5')
    inputs = keras.Input((None, None, 3))
    outputs = model(inputs)
    model = keras.models.Model(inputs, outputs)
    return model


@runway.command('upscale', inputs={'image': runway.image}, outputs={'image': runway.image})
def upscale(model, inputs):
    image = np.array(inputs['image']) / 255.0
    image = np.expand_dims(image, 0)
    out = model(image).numpy()
    out = ((out[0] + 1) * 127.5).astype(np.uint8)
    return out


if __name__ == "__main__":
    runway.run()