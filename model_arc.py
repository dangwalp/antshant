import tensorflow as tf
import numpy as np


def actv(inp):
    return tf.nn.leaky_relu(inp, alpha=0.01)


class ARCModel:
    def __init__(self, sample_rate, len_frame, seq_len):

        # Input, Output
        self.x_mixed = tf.placeholder(tf.float32, shape=(None, None, len_frame // 2 + 1), name='x_mixed')
        self.y_src1 = tf.placeholder(tf.float32, shape=(None, None, len_frame // 2 + 1), name='y_src1')
        self.y_src2 = tf.placeholder(tf.float32, shape=(None, None, len_frame // 2 + 1), name='y_src2')

        # Network
        self.seq_len = seq_len
        self.net = tf.make_template('net', self._net)
        self()
    
    def __call__(self):
        return self.net()

    def _net(self): 
        # Separation model (Fully Convolutional)
        epsilon = 0.0001

        # INPUT: (1025 x T) T=number of temporal frames
        spectro = self.x_mixed
        batch_size = tf.shape(spectro)[0]
       
        print("input", spectro.shape)
        print("batch_size", batch_size)

        # (output channels, filter size, stride)
        # Conv1: 1D convolution (1024, 3, 1)
        conv1 = actv(tf.layers.conv1d(inputs=spectro,
                                      filters=1024,
                                      kernel_size=3,
                                      strides=1,
                                      name='Conv1'))
        mean1, var1 = tf.nn.moments(conv1,[0])
        scale1 = tf.Variable(tf.ones([1024]))
        beta1 = tf.Variable(tf.zeros([1024]))
        conv1 = tf.nn.batch_normalization(conv1,mean1,var1,beta1,scale1,epsilon)
        print("conv1", conv1)

        # Conv2: 1D convolution (512, 3, 2) [skip GRU 1024 to TConv6]
        conv2 = actv(tf.layers.conv1d(inputs=conv1,
                                      filters=512,
                                      kernel_size=3,
                                      strides=2,
                                      name='Conv2'))
        mean2, var2 = tf.nn.moments(conv2,[0])
        scale2 = tf.Variable(tf.ones([512]))
        beta2 = tf.Variable(tf.zeros([512]))
        conv2 = tf.nn.batch_normalization(conv2,mean2,var2,beta2,scale2,epsilon)	
        print("conv2", conv2)

        # Conv3: 1D convolution (256, 3, 2) [skip GRU 512 to TConv5]
        conv3 = actv(tf.layers.conv1d(inputs=conv2,
                                      filters=256,
                                      kernel_size=3,
                                      strides=2,
                                      name='Conv3'))
        mean3, var3 = tf.nn.moments(conv3,[0])
        scale3 = tf.Variable(tf.ones([256]))
        beta3 = tf.Variable(tf.zeros([256]))
        conv3 = tf.nn.batch_normalization(conv3,mean3,var3,beta3,scale3,epsilon)
        print("conv3", conv3)

        # TConv4: 1D transposed convolution (512, 3, 2)
        filter4 = tf.Variable(tf.random_normal([3,512,256]))
        tconv4 = actv(tf.contrib.nn.conv1d_transpose(value=conv3,
                                                     filter= filter4,
                                                     output_shape = [batch_size, -1, 512],
                                                     stride=2,
                                                     padding="VALID",
                                                     name='TConv4'))
        mean4, var4 = tf.nn.moments(tconv4,[0])
        scale4 = tf.Variable(tf.ones([512]))
        beta4 = tf.Variable(tf.zeros([512]))
        tconv4 = tf.nn.batch_normalization(tconv4,mean4,var4,beta4,scale4,epsilon)
        print("conv4", tconv4)

        # TConv5: 1D transposed convolution (1024, 3, 2)
        
        # Transformation matrix
        self.W3 = tf.get_variable("W3", initializer=tf.random_normal(shape=(1,256,512), dtype=tf.float32)) 
        trans_conv3 = tf.matmul(conv3, self.W3, name="trans_conv3")
        
        # Filters
        filter5 = tf.Variable(tf.random_normal([3,1024,512]))
        tconv5 = actv(tf.contrib.nn.conv1d_transpose(
            value=tf.add(tconv4, trans_conv3),
            filter=filter5,
            output_shape = [batch_size, -1, 1024],
            stride=2,
            name='TConv5'))
        mean5, var5 = tf.nn.moments(tconv5,[0])
        scale5 = tf.Variable(tf.ones([1024]))
        beta5 = tf.Variable(tf.zeros([1024]))
        tconv5 = tf.nn.batch_normalization(tconv5,mean5,var5,beta5,scale5,epsilon)
        print("conv5", tconv5)

        # Conv6: 1D convolution (4*1025, 3, 1)
        
        # Transformation matrix
        self.W2 = tf.get_variable("W2", initializer=tf.random_normal(shape=(1,512,1024), dtype=tf.float32))
        trans_conv2 = tf.matmul(conv2, self.W2, name="trans_conv2")

        # Filters
        conv6 = actv(tf.layers.conv1d(
            inputs=tf.add(trans_conv2, tconv5),
            filters= 4*1025,
            kernel_size= 3,
            strides=1,
            name='Conv6'))
        print("conv6", conv6)

        #####
        # Enhancement model
        input_size = self.x_mixed.shape[2]
        y_hat_src1 = tf.layers.dense(inputs=conv3, units=input_size, activation=tf.nn.relu, name='y_hat_src1')
        y_hat_src2 = tf.layers.dense(inputs=conv3, units=input_size, activation=tf.nn.relu, name='y_hat_src2')

        # time-freq masking layer
        y_tilde_src1 = y_hat_src1 / (y_hat_src1 + y_hat_src2 + np.finfo(float).eps) * self.x_mixed
        y_tilde_src2 = y_hat_src2 / (y_hat_src1 + y_hat_src2 + np.finfo(float).eps) * self.x_mixed

        return y_tilde_src1, y_tilde_src2
        
    def loss(self):
        pred_y_src1, pred_y_src2 = self()
        return tf.reduce_mean(
            tf.square(self.y_src1 - pred_y_src1)
            + tf.square(self.y_src2 - pred_y_src2), name='loss')

    # @staticmethod
    # shape = (batch_size, n_freq, n_frames) => (batch_size, n_frames, n_freq)
    def spec_to_batch(self, src):
        num_wavs, freq, n_frames = src.shape
        
        # Padding
        pad_len = 0
        if n_frames % self.seq_len > 0:
            pad_len = (self.seq_len - (n_frames % self.seq_len))
        pad_width = ((0, 0), (0, 0), (0, pad_len))
        padded_src = np.pad(src, pad_width=pad_width, mode='constant',
                            constant_values=0)

        assert(padded_src.shape[-1] % self.seq_len == 0)

        batch = np.reshape(padded_src.transpose(0, 2, 1),
                           (-1, self.seq_len, freq))
        return batch, padded_src, freq

    @staticmethod
    def batch_to_spec(src, num_wav):
        # shape = (batch_size, n_frames, n_freq)
        #      => (batch_size, n_freq, n_frames)
        batch_size, seq_len, freq = src.shape
        src = np.reshape(src, (num_wav, -1, freq))
        src = src.transpose(0, 2, 1)
        return src

    @staticmethod
    def load_state(sess, ckpt_path):
        ckpt = tf.train.get_checkpoint_state(
            os.path.dirname(ckpt_path + '/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path:
            tf.train.Saver().restore(sess, ckpt.model_checkpoint_path)
