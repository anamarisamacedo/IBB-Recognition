import cv2
import numpy as np
import glob
import os
import json
from pathlib import Path
from scipy.spatial.distance import cdist
from shallowNN import ShallowNN
from preprocessing.preprocess import Preprocess
from metrics.evaluation_recognition import Evaluation
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix,f1_score,accuracy_score

# https://colab.research.google.com/github/bentrevett/pytorch-image-classification/blob/master/5_resnet.ipynb
# https://github.com/frederick0329/Image-Classification
class EvaluateAll:

    def __init__(self):
        os.chdir(os.path.dirname(os.path.realpath(__file__)))

        with open('config_recognition.json') as config_file:
            config = json.load(config_file)

        self.images_path = config['images_path']
        self.annotations_path = config['annotations_path']

    def clean_file_name(self, fname):
        return fname.split('/')[1].split(' ')[0]

    def get_annotations(self, annot_f):
        d = {}
        with open(annot_f) as f:
            lines = f.readlines()
            for line in lines:
                (key, val) = line.split(',')
                # keynum = int(self.clean_file_name(key))
                d[key] = int(val)
        return d

    def accuracy(self, Y, Y_pred):
        """
        Y: vector of true value
        Y_pred: vector of predicted value
        """
        def _to_binary(x):
            return 1 if x > .5 else 0

        assert Y.shape[0] == 1
        assert Y.shape == Y_pred.shape
        Y_pred = np.vectorize(_to_binary)(Y_pred)
        acc = float(np.dot(Y, Y_pred.T) + np.dot(1 - Y, 1 - Y_pred.T))/Y.size
        return acc

    def run_evaluation(self):

        im_list = sorted(glob.glob(self.images_path + '*.png', recursive=True))
       
        iou_arr = []
        preprocess = Preprocess()
        eval = Evaluation()

        cla_d = self.get_annotations(self.annotations_path)
        
        # Change the following extractors, modify and add your own

        # Pixel-wise comparison:
        import feature_extractors.pix2pix.extractor as p2p_ext
        pix2pix = p2p_ext.Pix2Pix()

        # Local Binary Patterns
        import feature_extractors.lbp.extractor as lbp_ext
        lbp = lbp_ext.LBP()
        
        lbp_features_arr = []
        plain_features_arr = []
        y = []

        for im_name in im_list:
            # Read an image
            img = cv2.imread(im_name)
            print(im_name)
            y.append(cla_d['test/'+im_name.split('\\')[1]])

            # Apply some preprocessing here

            
            # Run the feature extractors            
            #plain_features = pix2pix.extract(img)
            #plain_features_arr.append(plain_features)
            lbp_features = lbp.extract(img)
            lbp_features_arr.append(lbp_features)

        Y_plain = cdist(lbp_features_arr, lbp_features_arr, 'jensenshannon')

        r1 = eval.compute_rank5(Y_plain, y)
        print('LBP Rank-1[%]', r1)


    def run_evaluation_with_NN(self):

        im_list = sorted(glob.glob(self.images_path + '*.png', recursive=True))
        im_list_train = sorted(glob.glob("data/ear_detection_predictions/train/"+ '*.png', recursive=True))
       
        iou_arr = []
        preprocess = Preprocess()
        eval = Evaluation()

        cla_d = self.get_annotations(self.annotations_path)
        
        # Change the following extractors, modify and add your own

        # Local Binary Patterns
        import feature_extractors.lbp.extractor as lbp_ext
        lbp = lbp_ext.LBP()
        
        lbp_features_arr_test = []
        lbp_features_train_arr = []
        y_train = []
        y_test = []

        for im_name in im_list_train:
            # Read an image
            img = cv2.imread(im_name)
            y_train.append(cla_d['train/'+im_name.split('\\')[1]])
            
            # Apply some preprocessing here

            lbp_features_train = lbp.extract(img)
            lbp_features_train_arr.append(lbp_features_train)

        model = ShallowNN(20, 10, 1)
        model.fit(lbp_features_train_arr, y_train, batch_size=100, n_iterations=200, lr=0.01)

        for im_name in im_list:
            # Read an image
            img = cv2.imread(im_name)
            y_test.append(cla_d['test/'+im_name.split('\\')[1]])

            # Apply some preprocessing here
            
            # Run the feature extractors            
            lbp_features_test = lbp.extract(img)
            lbp_features_arr_test.append(lbp_features_test)
        
        y_preds = model.predict(lbp_features_arr_test)
        acc = self.accuracy(y_test.reshape(1, -1), y_preds)
        print(f'accuracy: {acc*100}%')
        #r1 = eval.compute_rank1(Y_plain, y)
        #print('Pix2Pix Rank-1[%]', r1)

    def run_evaluation_with_SVM(self):

        im_list = sorted(glob.glob(self.images_path + '*.png', recursive=True))
        im_list_train = sorted(glob.glob("data/ear_detection_predictions/train/"+ '*.png', recursive=True))
       
        iou_arr = []
        preprocess = Preprocess()
        eval = Evaluation()

        cla_d = self.get_annotations(self.annotations_path)
        
        # Change the following extractors, modify and add your own

        # Local Binary Patterns
        import feature_extractors.lbp.extractor as lbp_ext
        lbp = lbp_ext.LBP()
        
        lbp_features_arr_test = []
        lbp_features_train_arr = []
        y_train = []
        y_test = []

        for im_name in im_list_train:
            # Read an image
            img = cv2.imread(im_name)
            y_train.append(cla_d['train/'+im_name.split('\\')[1]])
            
            # Apply some preprocessing here

            lbp_features_train = lbp.extract(img)
            lbp_features_train_arr.append(lbp_features_train)

        from sklearn.svm import LinearSVC
        model = LinearSVC(max_iter=1000)
        model.fit(lbp_features_train_arr, y_train)

        for im_name in im_list:
            # Read an image
            img = cv2.imread(im_name)
            y_test.append(cla_d['test/'+im_name.split('\\')[1]])

            # Apply some preprocessing here
            
            # Run the feature extractors            
            lbp_features_test = lbp.extract(img)
            lbp_features_arr_test.append(lbp_features_test)
        
        y_preds = model.predict(lbp_features_arr_test)
        
        Y_plain = cdist(lbp_features_arr_test, lbp_features_arr_test, 'jensenshannon')
        r1 = eval.compute_rank1(Y_plain, y_preds)
        
        print('LBP Rank-1[%]', r1)

if __name__ == '__main__':
    ev = EvaluateAll()
    ev.run_evaluation()