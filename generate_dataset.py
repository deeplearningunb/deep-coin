import os
import shutil
import dlib
import cv2
import numpy as np


"""
Script to generate the dataset with the cutted coins images
"""

DATASETS_PATH = "datasets"
OUTPUT_FOLDER_PATH = DATASETS_PATH + "/generated_datasets"


def get_image_region(image, x1, y1, x2, y2):
    """
    Return image subregion defined by (x1, y1) and (x2, y2) rectangle.
    """

    return image[y1:y2, x1:x2]

def create_dataset(input_dataset_folder):
    """
    This script detect, cut and store coins from images.
    It's necessary to especify the directory path (train or test) where all images are stored.
    The directory should contain sub-directories as follow: '5', '10', '25', '50', '100'.

    arg:
    - input_dataset_folder: folder in datasets with inner folder of train (training) or test.
    """
    
    # Coin Detector made with dlib library
    detector = dlib.simple_object_detector("assets/coin_detector.svm")
    
    # Create output folder to store dataset with 64x64 coins   
    input_path = DATASETS_PATH + '/' + input_dataset_folder
    output_folder = OUTPUT_FOLDER_PATH + '/' + input_dataset_folder
    if os.path.exists(f'{output_folder}'):
        shutil.rmtree(f'{output_folder}')
    else:
        os.makedirs(f'{output_folder}')
    
    # Iterate every specific type of coin folder and create new dataset
    for coin_directory in os.listdir(f'{input_path}'):
        # Create directory to specific type of coin
        os.makedirs(f'{output_folder}/{coin_directory}')
        
        coin_images = []
        for coin_image in os.listdir(f'{input_path}/{coin_directory}'):
            # Read image from disk and detect coin
            img = cv2.imread(f'{input_path}/{coin_directory}/{coin_image}')
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            blur = cv2.GaussianBlur(gray, (3,3), 2)
            coin_image = detector(blur)

            # If some coin was found, append it to coin_images list
            if len(coin_image) > 0:
                left, top = coin_image[0].left(), coin_image[0].top()
                right, bottom = coin_image[0].right(), coin_image[0].bottom()
                coin_region = get_image_region(img, left, top, right, bottom)
                coin_region = np.array(coin_region)
                
                if (coin_region.shape[0] != 0 and coin_region.shape[1] != 0 and coin_region.shape[2] != 0):
                    coin_region = cv2.resize(coin_region,(64,64))
                    coin_images.append(coin_region)

        # Write coins images (64x64) to disk
        for index, image in enumerate(coin_images):
            cv2.imwrite(f'{output_folder}/{coin_directory}/{index}.jpg', image)


def main():
    print("Informe a pasta onde esta localizado o dataset com as duas pastas internas (train e test)")
    print("Obs: A pasta do dataset deve estar em /datasets (nao digite '/' no inicio ou fim do caminho)")
    input_dataset_folder = input()
    create_dataset(input_dataset_folder + '/train')
    create_dataset(input_dataset_folder + '/test')

if __name__ == '__main__':
    main()
