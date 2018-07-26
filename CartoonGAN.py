from networks import *
from utils import *
import tensorflow as tf
import numpy as np
from PIL import Image
import os


batchsize = 5
w = 10
img_w = 128
img_h = 128


class CartoonGAN:
    def __init__(self):
        self.p = tf.placeholder("float", [batchsize, img_h, img_w, 3])
        self.e = tf.placeholder("float", [batchsize, img_h, img_w, 3])
        self.c = tf.placeholder("float", [batchsize, img_h, img_w, 3])
        self.G = generator("generator")
        self.D = discriminator("discriminator")
        vgg = VGG("VGG19")
        self.fake_img = self.G(self.p)
        #label: 0:c, 1:e, 2:g
        self.L_adv = tf.reduce_mean(tf.log(self.D(self.c) + epsilon)) + \
                     tf.reduce_mean(tf.log(1 - self.D(self.e, True) + epsilon)) + \
                     tf.reduce_mean(tf.log(1 - self.D(self.fake_img, True) + epsilon))
        self.L_con = tf.reduce_mean(tf.abs(vgg(self.fake_img) - vgg(self.p)))
        self.D_loss = - self.L_adv
        self.G_loss = - tf.reduce_mean(tf.log(self.D(self.fake_img, True) + epsilon)) + w * self.L_con
        self.D_Opt = tf.train.AdamOptimizer(2e-4).minimize(self.D_loss, var_list=self.D.var)
        self.G_Opt = tf.train.AdamOptimizer(2e-4).minimize(self.G_loss, var_list=self.G.var)
        self.G_init_Opt = tf.train.AdamOptimizer(1e-3).minimize(self.L_con, var_list=self.G.var)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def train(self, is_init=True):
        path_p = "E://DeepLearn_Experiment//MSCOCO//"
        path_c = "E://DeepLearn_Experiment//CartoonSet//c//"
        path_e = "E://DeepLearn_Experiment//CartoonSet//e//"
        filenames_p = os.listdir(path_p)
        filenames_c = os.listdir(path_c)
        filenames_e = os.listdir(path_e)
        saver = tf.train.Saver(var_list=self.G.var)
        if not is_init:
            for epoch in range(10):
                for i in range(filenames_c.__len__()//batchsize - 1):
                    batch_c = np.zeros([batchsize, img_h, img_w, 3])
                    batch_p = np.zeros([batchsize, img_h, img_w, 3])
                    for j in np.arange(i*batchsize, i*batchsize+batchsize):
                        batch_c[j-i*batchsize, :, :, :] = resize_and_crop(np.array(Image.open(path_c+filenames_c[j])), img_h)
                        batch_p[j-i*batchsize, :, :, :] = resize_and_crop(np.array(Image.open(path_p + filenames_p[j])), img_h)
                    [_, loss] = self.sess.run([self.G_init_Opt, self.L_con], feed_dict={self.p: batch_p, self.c: batch_c})
                    if i % batchsize == 0:
                        print("Epoch: %d, step: %d, content_loss: %f"%(epoch, i, loss))
                        fake_img = self.sess.run(self.fake_img, feed_dict={self.p: batch_p, self.c: batch_c})
                        Image.fromarray(np.uint8(fake_img[0, :, :, :])).save("./results/"+str(i)+".jpg")
                saver.save(self.sess, "./init_generator_para/init.ckpt")
        saver.restore(self.sess, "./init_generator_para/init.ckpt")
        saver = tf.train.Saver()
        for epoch in range(200):
            for i in range(filenames_c.__len__() // batchsize - 1):
                batch_c = np.zeros([batchsize, img_h, img_w, 3])
                batch_e = np.zeros([batchsize, img_h, img_w, 3])
                batch_p = np.zeros([batchsize, img_h, img_w, 3])
                for j in np.arange(i * batchsize, i * batchsize + batchsize):
                    batch_c[j - i * batchsize, :, :, :] = resize_and_crop(np.array(Image.open(path_c + filenames_c[j])), img_h)
                    batch_e[j - i * batchsize, :, :, :] = resize_and_crop(np.array(Image.open(path_e + filenames_e[j])), img_h)
                    batch_p[j - i * batchsize, :, :, :] = resize_and_crop(np.array(Image.open(path_p + filenames_p[j])), img_h)
                self.sess.run(self.D_Opt, feed_dict={self.p: batch_p, self.c: batch_c, self.e: batch_e})
                self.sess.run(self.G_Opt, feed_dict={self.p: batch_p, self.c: batch_c, self.e: batch_e})
                if i % batchsize == 0:
                    [Dloss, Gloss] = self.sess.run([self.D_loss, self.G_loss], feed_dict={self.p: batch_p, self.c: batch_c, self.e: batch_e})
                    print("Epoch: %d, step: %d, D_loss: %f, G_loss: %f" % (epoch, i, Dloss, Gloss))
                    fake_img = self.sess.run(self.fake_img, feed_dict={self.p: batch_p, self.c: batch_c})
                    Image.fromarray(np.uint8(fake_img[0, :, :, :])).save("./results/" + str(i) + ".jpg")
            saver.save(self.sess, "./para/init.ckpt")
            pass



if __name__ == "__main__":
    CGAN = CartoonGAN()
    CGAN.train(True)
