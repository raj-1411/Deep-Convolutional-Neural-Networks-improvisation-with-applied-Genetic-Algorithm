from Feature_extractors import Extractor
from GA import base
import argparse



parser = argparse.ArgumentParser(description = 'Application of Genetic Algorithm')
# Paths
parser.add_argument('-data','--data_path',type=str, 
                    default = 'data', 
                    help = 'Path to data')
parser.add_argument('-classes','--num_classes',type=int, 
                    default = 2, 
                    help = 'Number of data classes')
parser.add_argument('-ext','--ext_type',type=str, 
                    default = 'resnet', 
                    help = 'Choice of extractor')                    
parser.add_argument('-classif','--classif_type',type=str, 
                    default = 'MLP', 
                    help = 'Choice of classifier for GA')




args = parser.parse_args()
folder_path = args.data_path
out_classes = args.num_classes
ext = args.ext_type
classif = args.classif_type


print("Extracting features...")
print('\n'*2)
arr_tr, arr_val = Extractor.featr_ext(folder_path, ext, out_classes)
print("Features extracted.")
print('\n'*4)
print('Starting Genetic Algorithm...')
print('\n'*4)
base.algo(arr_tr, arr_val, classif)
