"""
Converter Keras2Python
======================
"""
import argparse
import os
from tensorflow import keras


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'keras_model',
        help='keras model to convert')

    args = parser.parse_args()

    ipt = args.keras_model
    name = os.path.splitext(os.path.basename(ipt))[0]
    model = keras.models.load_model(ipt)
    otp = name + '.py'

    with open(otp, "w") as pythoncode:
        print("", file=pythoncode)
        l = 1
        for layer in model.layers:
            weights = layer.get_weights()[0]
            biases = layer.get_weights()[1]
            ins = layer.get_weights()[0].shape[0]
            outs = layer.get_weights()[0].shape[1]
            for i in range(0, outs):
                print("x%d%d =" % (l, i), end=" ", file=pythoncode)
                for j in range(0, ins):
                    print("(%f)*x%d%d +" % (weights[j][i], l - 1, j), end=" ", file=pythoncode)
                print("(%f)" % biases[i], file=pythoncode)
            # activation
            print("#", file=pythoncode)
            for i in range(0, outs):
                if layer.get_config().get('activation', None) == 'relu':
                    print("ReLU(x%d%d)" % (l, i), file=pythoncode)
            print("", file=pythoncode)
            l = l + 1


if __name__ == '__main__':
    main()
