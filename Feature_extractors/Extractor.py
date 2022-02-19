from Feature_extractors import GoogLeNet,ResNet18,VGG19
import numpy as np

def featr_ext(folder_path, ext_type, out_classes):

    if ext_type == 'googlenet':

        arr_train, labels_tr, arr_val, labels_val = GoogLeNet.extr(folder_path, out_classes)
        arr_train = np.append(arr_train[1:,:], np.array(labels_tr).reshape(-1,1), axis=1)
        arr_val = np.append(arr_val[1:,:], np.array(labels_val).reshape(-1,1), axis=1)
        return arr_train, arr_val

    elif ext_type == 'vgg':
        
        arr_train, labels_tr, arr_val, labels_val = VGG19.extr(folder_path, out_classes)
        arr_train = np.append(arr_train[1:,:], np.array(labels_tr).reshape(-1,1), axis=1)
        arr_val = np.append(arr_val[1:,:], np.array(labels_val).reshape(-1,1), axis=1)
        return arr_train, arr_val

    else:
        
        arr_train, labels_tr, arr_val, labels_val = ResNet18.extr(folder_path, out_classes)
        arr_train = np.append(arr_train[1:,:], np.array(labels_tr).reshape(-1,1), axis=1)
        arr_val = np.append(arr_val[1:,:], np.array(labels_val).reshape(-1,1), axis=1)
        return arr_train, arr_val