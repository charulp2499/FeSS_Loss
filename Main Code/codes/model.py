import tensorflow as tf
import numpy as np

class UNET3D(tf.keras.Model):
    def __init__(self, classes, dropout_rate=0.5, l2_reg=0.01):
        super(UNET3D, self).__init__()
        self.classes = classes
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg

    def conv3D(self, x, filters, filter_size, activation='relu'):
        out = tf.keras.layers.Conv3D(
            filters,
            (filter_size, filter_size, filter_size),
            padding="same",
            kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg)
        )(x)
        out = tf.keras.layers.BatchNormalization()(out)
        out = tf.keras.layers.Activation('relu')(out)
        out = tf.keras.layers.Dropout(self.dropout_rate)(out)
        return out

    def upsampling3D(self, x, filters, filter_size, stride=2, activation='relu'):
        up_out = tf.keras.layers.Conv3DTranspose(
            filters, (filter_size, filter_size, filter_size),
            strides=(stride, stride, stride), padding='same'
        )(x)
        out = tf.keras.layers.BatchNormalization()(up_out)
        return out

    def concatenate(self, x1, x2):
        concat = tf.keras.layers.concatenate([x1, x2])
        return concat

    def max_pool3D(self, x, filter_size, stride):
        out = tf.keras.layers.MaxPooling3D(
            (filter_size, filter_size, filter_size), strides=stride
        )(x)
        return out

    def downconv(self, x, filters):
        s1 = self.conv3D(x, filters, 3)
        s2 = self.conv3D(s1, filters, 3)
        return s1

    def upconv(self, x, filters, skip_connection):
        e1 = self.upsampling3D(x, filters, 2)
        concat = self.concatenate(e1, skip_connection)
        conv1 = self.conv3D(concat, filters, 3)
        conv2 = self.conv3D(conv1, filters, 3)
        return conv2

    def call(self):
        inputs = tf.keras.layers.Input(shape=(64,64,64, 1))
        d1 = self.downconv(inputs, 32)
        m1 = self.max_pool3D(d1, filter_size=2, stride=2)
        d2 = self.downconv(m1, 64)
        m2 = self.max_pool3D(d2, filter_size=2, stride=2)
        d3 = self.downconv(m2, 128)
        m3 = self.max_pool3D(d3, filter_size=2, stride=2)
        d4 = self.downconv(m3, 256)
        m4 = self.max_pool3D(d4, filter_size=2, stride=2)

        bridge = self.conv3D(m4, 1024, 3, 1)
        bridge = self.conv3D(bridge, 1024, 3, 1)

        u1 = self.upconv(bridge, 256, d4)
        u2 = self.upconv(u1, 128, d3)
        u3 = self.upconv(u2, 64, d2)
        u4 = self.upconv(u3, 32, d1)

        logits = tf.keras.layers.Conv3D(self.classes, (1, 1, 1), padding="same")(u4)
        logits = tf.nn.sigmoid(logits)

        model = tf.keras.Model(inputs=[inputs], outputs=[bridge, logits])
        return model