### Imports
import os
import requests
import cv2
from collections.abc import Iterable   # drop `.abc` with Python 2.7 or lower

def is_iterable(obj):
    try:
        iter(obj)
        return True
    except Exception as e:
        return False


### Parameters
API_URL = "http://54.216.71.72:5050/image"
ROOT_PATH = "..\\"
DATA_PATH = ROOT_PATH + "data_for_test"


### Functions
def call_api(image):
    files = {"fileImage": open(image, "rb")}
    r = requests.post(url, files=files)
    print(r)
    print(r.text)
    predict_str = r.text
    return(predict_str)

def test(input, expected_output):
    output = call_api(input)
    return is_iterable(expected_output) and output in expected_output or output == expected_output



### Main
if __name__ == "__main__":
    # Test with jpg
    image = cv2.imread(DATA_PATH + "jpg.jpg", cv2.IMREAD_UNCHANGED)
    test(image, )

    # Test with png

    # Test with webm

    # Test with gif

    # Test with mp3

    # Test with mp4

    # Test with nothing
    expected_output = ""
    res = test_call_api(None)

    # Test with text
    test_call_api("..\\models\\best_CNN_Conv32_MaxPool2_Conv64_MaxPool2_Conv32_MaxPool2_Dense64relu_Dropout05_Dense32Relu_Dropout05_Dense6Relu.h5")

    # Test with int
    test_call_api(123456)

    # Test with float
    test_call_api(789.101112)

    # Test with random bites
    test_call_api(urandom(8))

    # Testing simultaneous call
