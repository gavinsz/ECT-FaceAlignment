# -*- coding: UTF-8 -*-

import matplotlib
matplotlib.use('Agg')
import argparse
import sys
import os, os.path
import pickle
import scipy
import matplotlib.pyplot as plt
import numpy as np
from menpo.visualize import print_progress
from pylab import *

import menpo.io as mio
from menpofit.clm import CLM, FcnFilterExpertEnsemble
from menpofit.clm import GradientDescentCLMFitter
from menpo.visualize import print_dynamic
from menpo.shape.pointcloud import PointCloud
from menpofit.fitter import noisy_shape_from_bounding_box
from menpofit.error import euclidean_distance_indexed_normalised_error
from menpofit.error import euclidean_distance_normalised_error
from menpofit.error import inner_pupil
from PIL import Image
import rspimage
import Queue
import threading
import time
from multiprocessing import Manager,Pool

# initial the caffe net
os.environ['GLOG_minloglevel'] = '1'
#caffe_root = '../caffe/'
caffe_root = '/data/home/chenxinliu/caffe-master/'
sys.path.insert(0, caffe_root + 'python')
import caffe

def get_center(point1, point2):
    #print point1, point2
    cent = [(point1[0]+point2[0])/2, (point1[1]+point2[1])/2]    
    #print 'center', cent
    return cent

def gen_5pt(points):
    a = np.array([points[33], get_center(points[36], points[39]), get_center(points[42], points[45]), points[48], points[54]])
    return a/2

def main(args, q, lock):

    if args.gpus != None:
        caffe.set_device(args.gpus)
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()

    # load FCN model
    global net
    lock.acquire()
    net = caffe.Net(args.prototxt, args.model, caffe.TEST)
    lock.release()

    global transformer
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
    transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
    #transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR
    global fit_model
    # load PDM model
    fit_model_file = open('PDM_300w.pic', 'r')
    fit_model = pickle.load(fit_model_file)
    fit_model_file.close()

    # parameters for weighted regularized mean-shift
    fit_model.opt = dict()
    fit_model.opt['numIter'] = args.nIter
    fit_model.opt['kernel_covariance'] = 10
    fit_model.opt['sigOffset'] = 25
    fit_model.opt['sigRate'] = 0.25
    fit_model.opt['pdm_rho'] = 20
    fit_model.opt['verbose'] = args.verbose
    fit_model.opt['rho2'] = 20
    fit_model.opt['dataset'] = 'demo'
    fit_model.opt['ablation'] = (True, True)
    fit_model.opt['ratio1'] = 0.12
    fit_model.opt['ratio2'] = 0.08
    fit_model.opt['imgDir'] = args.imgDir
    fit_model.opt['smooth'] = True
    
    global fitter
    fitter = GradientDescentCLMFitter(fit_model, n_shape=[args.nComponent])

    print args.imgDir  
    
    for i in range(q.qsize()):
        img_dir = q.get()
        extract_features(img_dir, lock)
    
    '''
    dir_list = get_img_dir_list(args.imgDir)
    for img_dir in dir_list:
        
        ## 填充队列
        #queueLock.acquire()
        #workQueue.put(img_dir)
        #print 'put ', img_dir
        #print 'main() q.qsize=', q.qsize()
        #queueLock.release()
        #print '@@@@@@@queueLock release'
    print 'main() q.qsize=', q.qsize() 
    '''
    
def get_img_dir_list(path):
    s = []
    g = os.walk(path)
    for path, dir_list, file_list in g:
        #print 'dir_list:\n', dir_list
        #print 'file_list:\n', file_list
        for dir_name in dir_list:
            #print os.path.join(path, dir_name)
            s.append(os.path.join(path, dir_name))

    return s


exitFlag = 0
class myThread (threading.Thread):
    def __init__(self, threadID, name, q):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.q = q
    def run(self):
        print "Starting " + self.name
        process_data(self.name, self.q)
        print "Exiting " + self.name

def process_data(q, name):
    print 'q.qsize=', q.qsize
    for i in range(q.qsize()):
        img_dir = q.get()
        #print img_dir
        extract_features(img_dir)

    '''
    while not exitFlag:
        queueLock.acquire()
        print threadName,' lock acquried succ'
        if not workQueue.empty():
            img_dir = workQueue.get()
            print 'get ', img_dir
            queueLock.release()
            extract_features(img_dir)
        else:
            print 'workQueue is empty'
            queueLock.release()
        time.sleep(1)
    '''

def extract_features(img_dir, lock):
    print 'load images from ', img_dir
    p2pErrs = []
    fitting_results = []
    indexCount = 0
    imageList = mio.import_images(img_dir, verbose=True)
    indexAll = len(imageList)
    for i in imageList:
        # input images with size of 256x256
        if i.shape[0] != i.shape[1] or i.shape[0] != 256:
            zoomImg = scipy.ndimage.zoom(i.pixels, zoom=[1, 256 / float(i.shape[1]), 256 / float(i.shape[1])])
            i.pixels = zoomImg
        # check whether the ground-truth is provided or not
        try:
            i.landmarks['PTS']
        except:
            i.landmarks['PTS'] = fit_model.reference_shape
        # Estimation step, get response maps from FCN
        net.blobs['data'].data[...] = transformer.preprocess('data', np.rollaxis(i.pixels, 0, 3))

        i.rspmap_data = np.array(net.forward()['upsample'])
        # zoom response maps
        # i.rspmap_data = scipy.ndimage.zoom(i.rspmap_data, zoom=[1, 1, float(i.height) / i.rspmap_data.shape[-2],
        #                                                               float(i.width) / i.rspmap_data.shape[-1]], order=1)  # mode = 'nearest'

        gt_s = i.landmarks['PTS'].lms
        s = rspimage.initial_shape_fromMap(i)
        # fit image
        fr = fitter.fit_from_shape(i, s, gt_shape=gt_s)
        
        fitting_results.append(fr)
        # calculate point-to-point Normalized Mean Error
        Err = euclidean_distance_normalised_error(fr.shapes[-1], fr.gt_shape, distance_norm_f=inner_pupil)

        p2pErrs.append(Err)

        text_file = open(img_dir + '/' + i.path.stem + '.68pt', "w")
        np.savetxt(text_file, fr.shapes[-1].points, fmt='%d', newline='\n')

        print img_dir + '/' + i.path.stem + '.5pt'
        five_pt_text_file = open(img_dir + '/' + i.path.stem + '.5pt', "w")
        five_pt_array = gen_5pt(fr.shapes[-1].points)
        np.savetxt(five_pt_text_file, five_pt_array, fmt='%d', newline='\n')
        
        text_file.close()
        five_pt_text_file.close()
        
        indexCount = indexCount + 1
        # sys.stdout.write('{} done;'.format(i.path.name))
        sys.stdout.write('\r')
        sys.stdout.write('{}/{} Done; '.format(indexCount,indexAll))
        sys.stdout.flush()

    p2pErrs = np.array(p2pErrs)
    print('NormalizedMeanError: {:.4f}'.format(average(p2pErrs)))


threadList = []
nameList = ["One", "Two", "Three", "Four", "Five"]
queueLock = threading.Lock()
workQueue = Queue.Queue(90000)
threads = []
threadID = 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ECT for face alignment')

    parser.add_argument('--gpus', default=None, type=int, help='specify the gpu ID')
    parser.add_argument('--imgDir', default='../imgs/', type=str, help='path to test images')
    parser.add_argument('--outDir', default='../output/', type=str, help='path for saving prediction results')
    parser.add_argument('--prototxt', default='../caffe/models/300w/matlab.prototxt', type=str,
                        help='path to caffe model prototxt')
    parser.add_argument('--model', default='../model_data/300w_68pt.caffemodel', type=str,
                        help='path to the pre-trained caffe model')
    parser.add_argument('--verbose', default=True, help='show the landmark prediction results')
    parser.add_argument('--nIter', default=5, type=int, help='number of iterations for the turning step')
    parser.add_argument('--nComponent', default=30, type=int, help='number of PDM components to be used')
    
    q = Manager().Queue()  
    po = Pool()
    lock = Manager().Lock()
    
    #for i in range(0, 20):
    #    threadList.append('Thread_' + str(i))
    #    
    ## 创建新线程
    #for tName in threadList:
    #    thread = myThread(threadID, tName, workQueue)
    #    thread.start()
    #    threads.append(thread)
    #    threadID += 1
    #
    #main(parser.parse_args(), q)
    
    dir_list = get_img_dir_list(parser.parse_args().imgDir)
    args = parser.parse_args()

    length = len(dir_list)
    if (0 == args.gpus):
        start = 0
        end = length / 2
    else:
        start = length / 2 - 1
        end = length

    for i in range(start, end):
        q.put(dir_list[i])
        
    #for img_dir in dir_list:
    #    q.put(img_dir)
    
    print 'q.qsize=', q.qsize()
    
    for i in range(0, 5):
        po.apply_async(main, args=(parser.parse_args(), q, lock))

    # 等待队列清空
    while not workQueue.empty():
        pass
    
    while not q.empty():
        pass

    po.close()
    po.join()
    # 通知线程是时候退出
    exitFlag = 1
     
    # 等待所有线程完成
    #for t in threads:
    #    t.join()

    print "Exiting Main Thread"

    
