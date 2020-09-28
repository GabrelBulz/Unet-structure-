import os 
import numpy as np
import cv2
import model
from tensorflow.keras.callbacks import ModelCheckpoint
from data_generator import DataGenerator




def process_img(img):
    return img/255

def process_mask(mask):
    mask = mask/255
    mask[mask >= 0.5] = 1
    mask[mask < 0.5] = 0
    return mask

def create_daset(path_img, path_mask):
    file_names = os.listdir(path_img)

    for img_name in file_names:
        img = cv2.imread(path_img + '\\' + img_name)
        img = process_img(img)
        mask = cv2.imread(path_mask + '\\' + img_name, 0)
        mask = process_mask(mask)
        pair = img, mask
        yield pair

def create_img_and_mask_paths(path_img, path_mask):
    img_paths = []
    mask_paths = [] 

    name = os.listdir(path_img)
    for i in name:
        img_paths.append(path_img + '\\' + i)
        mask_paths.append(path_mask + '\\' + i)

    return img_paths, mask_paths


def main():
    PATH_IMG = 'C:\\Users\\bulzg\\Desktop\\road_detection\\train_set\\img'
    PATH_MASK = 'C:\\Users\\bulzg\\Desktop\\road_detection\\train_set\\mask_line'
    PAHT_MODEL = 'C:\\Users\\bulzg\Desktop\\road_detection\\train_set'
    os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

    # dataset = create_daset(PATH_IMG, PATH_MASK)

    # xx = None 
    # yy = None

    img_paths, mask_paths = create_img_and_mask_paths(PATH_IMG, PATH_MASK)

    generator = DataGenerator(img_paths, mask_paths, 1, True)
    # print(len(generator))

    # print(len(dataset))

    # for i,j in dataset:
    #     xx = i
    #     yy = j
    #     break

    model_name = 'unet_weights'
    model_unet = model.unet(1, (512,512,3), model_name, False, 4)
    
    checkpoint = ModelCheckpoint(os.path.join(PAHT_MODEL, model_name+'.model'), monitor='dice', verbose=1, mode='max',
                             save_best_only=True, save_weights_only=False, period=10)

    # xx = xx.reshape(-1,512,512,3)
    # print(xx.shape)
    # return
    # model_unet.fit(x=xx, y=yy,
    #                 steps_per_epoch=300,
    #                 epochs=20, verbose=1,
    #                 callbacks=[checkpoint])

    model_unet.fit_generator(generator=generator,
                    steps_per_epoch=len(generator),
                    epochs=300, verbose=1,
                    callbacks=[checkpoint])


main()
