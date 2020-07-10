import cv2.cv2 as cv2
import os
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.python.keras.models import load_model


def processing(input_path: str,
               croping: bool = True,
               resizing: bool = True,
               blurring: bool = True):
    """ Preproccesed an image to be model compliant

    :param input_path: str
        Path of the image to be processed.
    :param output_path: str
        Path to write the resulting image
    :param croping: bool
        If True, will crop few pixels line at the bottom of the image to remove potential watermark.
    :param resizing: bool
        If True, will resize the image in a 50 by 50 pixels format.
    :param blurring: bool
        If True, will slightly blur the image to remove potential noise.

    :return: None
    """
    width = 50
    height = 50
    new_dimensions = (width, height)
    # image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    image = np.array(input_path)

    if image.shape[2] != 3:
        image = np.delete(image, 3, 2)


    if croping:
        croped_image = image[0:int(image.shape[1] * 0.9), 0:]
    else:
        croped_image = image

    if resizing:
        image_resized = cv2.resize(croped_image,
                                   new_dimensions,
                                   interpolation=cv2.INTER_LINEAR)
    else:
        image_resized = croped_image

    if blurring:
        blurred_image = cv2.GaussianBlur(image_resized, (5, 5), 0)
    else:
        blurred_image = image_resized

    return blurred_image


def old_processing(img_path):
    # img = image.load_img(img_path, target_size=(300, 300))
    img = img_path.resize((300,300),0,None)
    img = image.img_to_array(img, dtype=np.uint8)
    img = np.array(img) / 255
    return img


def model_predict(model, image):
    class_val = {
        'verre': 1,
        'papier': 2,
        'carton': 3,
        'plastique': 4,
        'metal': 5,
    }
    p = model.predict(image[np.newaxis, ...])
    model_val = [np.argmax(p[0], axis=-1)]
    try:
        for key, value in class_val.items():
            # print(f" cle : {key} , value : {value} \n")
            if model_val[0] == value:
                return f"{key} : {np.max(p[0], axis=-1)}"
    except Exception as e:
        print(e)
        return ('Error on image processiong')


if __name__ == "__main__":
    pass
