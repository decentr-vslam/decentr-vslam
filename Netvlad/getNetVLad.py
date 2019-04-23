import cv2
import os
import glob
import json
import numpy as np
import tensorflow as tf
import Netvlad.nets as nets


class ImgDesc(object):
    def __init__(self, is_grayscale=True):
        self.is_grayscale = is_grayscale
        if is_grayscale:
            self.tf_batch = tf.placeholder(
                    dtype=tf.float32, shape=[None, None, None, 1])
        else:
            self.tf_batch = tf.placeholder(
                    dtype=tf.float32, shape=[None, None, None, 3])
        self.net_out = nets.vgg16NetvladPca(self.tf_batch)
        saver = tf.train.Saver()
        self.sess = tf.Session()
        saver.restore(self.sess, nets.defaultCheckpoint())



    def Getdescrpt(self, path, batch_size, verbose=False):
        '''
        :param path: the path of dataset
        :param batch_size: batch size
        :param verbose: s
        :param save:
        :return: a list of descriptions
        '''
        jpeg_paths = sorted(glob.glob(os.path.join(path, '*.jpg')))
        descriptions = []
        for batch_offset in range(0, len(jpeg_paths), batch_size):
            images = []
            for i in range(batch_offset, batch_offset + batch_size):
                if i == len(jpeg_paths):
                    break
                if verbose:
                    print('%d/%d' % (i, len(jpeg_paths)))
                if self.is_grayscale:
                    image = cv2.imread(jpeg_paths[i], cv2.IMREAD_GRAYSCALE)
                    images.append(np.expand_dims(
                            np.expand_dims(image, axis=0), axis=-1))
                else:
                    image = cv2.imread(jpeg_paths[i])
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    images.append(np.expand_dims(image, axis=0))
            batch = np.concatenate(images, 0)
            desc = list(self.sess.run(self.net_out, feed_dict={self.tf_batch: batch}))
            descriptions = descriptions + desc
        return descriptions


    def Savedescrpt(self, path, save_dir, verbose=True):
        '''
        :param path: the path of dataset
        :param save_dir:
        :param verbose:
        :return:
        '''
        jpeg_paths = sorted(glob.glob(os.path.join(path, '*.jpg')))
        file = open(save_dir,'w')
        for num, path in enumerate(jpeg_paths):
            file_name = '{:06d}.jpg'.format(num)
            if verbose:
                print('%d/%d' % (num, len(jpeg_paths)))
            if self.is_grayscale:
                image = cv2.imread(jpeg_paths[num], cv2.IMREAD_GRAYSCALE)
                image = np.expand_dims(
                        np.expand_dims(image, axis=0), axis=-1)
            else:
                image = cv2.imread(jpeg_paths[num])
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = np.expand_dims(image, axis=0)
            desc = self.sess.run(self.net_out, feed_dict={self.tf_batch: image})
            dic={
                'file_name': file_name,
                'descriptor': desc.tolist()}
            json.dump(dic,file)
            file.write('\n')
        return




tf.reset_default_graph()
imd = ImgDesc(is_grayscale=True)
dataset_path = '/Users/luciawen/GitLab/dslam/Netvlad/kitti/00/image_0/'
save_dir = '/Users/luciawen/GitLab/dslam/Netvlad/netvlad.json'
feats = imd.Savedescrpt(dataset_path,save_dir, verbose=True)
print('end')