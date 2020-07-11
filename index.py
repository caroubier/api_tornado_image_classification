import tornado.web
import tornado.ioloop
from call_model import *
import io
from PIL import Image
import uuid
from os.path import dirname, abspath
import sys

class ImgClassifierHandler(tornado.web.RequestHandler):
    def post(self):
        valid_ext = ['JPG', 'JPEG', 'PNG']

        try:
            file1 = self.request.files["fileImage"][0]
            file_name = file1['filename']
            file_ext = os.path.splitext(file_name)[1]
            file_body = file1['body']
        except Exception as e:
            print(e)
            self.write(f"POST request empty!")
            self.finish()
            return

        try:
          img = io.BytesIO(file_body)
        except Exception as e:
            print(e)
            self.write(f"Unable to read file!")
            self.finish()
            return

        file_ext = file_ext.strip().upper().replace(".", "")

        if file_ext not in valid_ext:
            self.write(f"only jpg, jpeg and png are ok")
            self.finish()
            return
        try:
            img = Image.open(img)
        except Exception as e:
            print(e)
            self.write(f"unable to open file !")
            self.finish()
        try:
            if CHOSE_PROCESSING == 'old':
                processed_img = old_processing(img)
            else:
                processed_img = processing(img)
        except Exception as e:
            print(e)
            self.write(f"unable to process your file !")
            self.finish()

        try:
            result = model_predict(model, processed_img)
            print(result)
            if result:
                self.write(f"{result}")
            else:
                self.write("Model unable to process file !")
        except Exception as e:
            print(e)
            self.write(f"impossible to put file in model !")
            self.finish()




class IndexHandler(tornado.web.RequestHandler):
    def get(self):
        self.render('index.html')


class VsdHandler(tornado.web.RequestHandler):
    def post(self):
        valid_ext = ['JPG', 'JPEG', 'PNG']


        try:
            file1 = self.request.files["fileImage"][0]
            value = self.get_argument('model_class')
            user_validation = self.get_argument('user_validation')
            file_name = file1['filename']
            file_ext = os.path.splitext(file_name)[1]
            file_body = file1['body']

        except Exception as e:
            print(e)
            self.write(f"POST request empty!")
            self.finish()
            return

        file_ext = file_ext.strip().upper().replace(".", "")

        if file_ext not in valid_ext:
            self.write(f"only jpg, jpeg and png are ok")
            self.finish()
            return

        try:
            img = io.BytesIO(file_body)
            img = Image.open(img)
        except Exception as e:
            print(e)
            self.write(f"Unable to read file!")
            self.finish()
            return

        if user_validation == "1":
            fname = "user_"+str(uuid.uuid4())+"."+file_ext.lower()
        else:
            fname = str(uuid.uuid4()) + "." + file_ext.lower()

        image_directory = ""

        for cle, valeur in CLASS_VAL.items():
            if str(valeur) == value:
                image_directory = cle
                break

        image_folder = os.path.join(PARENT_PATH,SAMPLES_DATASET, image_directory, fname)

        try:
            img.save(image_folder)
        except:
            self.write(f"Unable to keep file!")
            self.finish()

        self.write(f"thank you for your feedback !")


if __name__ == "__main__":
    # PORT = 5050
    # CHOSE_PROCESSING = "old"
    # model_name = "best_CNN_Conv32_MaxPool2_Conv64_MaxPool2_Conv32_MaxPool2_Dense64relu_Dropout05_Dense32Relu_Dropout05_Dense6Relu.h5"
    PORT = int(sys.argv[1])
    CHOSE_PROCESSING = sys.argv[2] # old : old_processing ; new : new_processing
    model_name = sys.argv[3] # model name ; has to be stored in output-training/modeles/

    # ex de commande
    #  python index.py 5050 "old" "best_CNN_Conv32_MaxPool2_Conv64_MaxPool2_Conv32_MaxPool2_Dense64relu_Dropout05_Dense32Relu_Dropout05_Dense6Relu.h5"

    ROOT_PATH = os.getcwd()
    PARENT_PATH = dirname(os.getcwd())
    SAMPLES_DATASET = "dataset-trashnet"
    OUTPUT_TRAINING = "output-training"
    MODEL_FOLDER = "modeles"
    LOGS = "logs"
    CLASS_VAL = {
        'verre': 1,
        'papier': 2,
        'carton': 3,
        'plastique': 4,
        'metal': 5,
    }

    if not os.path.exists(os.path.join(PARENT_PATH,OUTPUT_TRAINING, LOGS)):
        os.makedirs(os.path.join(PARENT_PATH,OUTPUT_TRAINING, LOGS))

    if not os.path.exists(os.path.join(PARENT_PATH,OUTPUT_TRAINING, MODEL_FOLDER)):
        os.makedirs(os.path.join(PARENT_PATH,OUTPUT_TRAINING, MODEL_FOLDER))

    if not os.path.exists(os.path.join(PARENT_PATH,SAMPLES_DATASET)):
        os.makedirs(os.path.join(PARENT_PATH,SAMPLES_DATASET))

    for key, value in CLASS_VAL.items():
        if not os.path.exists((os.path.join(PARENT_PATH, SAMPLES_DATASET, key))):
            os.mkdir(os.path.join(PARENT_PATH, SAMPLES_DATASET, key))

    model_path = os.path.join(PARENT_PATH,OUTPUT_TRAINING,MODEL_FOLDER)

    model = load_model(os.path.join(model_path, model_name))

    app = tornado.web.Application([
        (r"/image", ImgClassifierHandler),
        (r"/uvsd", VsdHandler), # user  validated sample dataset / response validated by user
        (r"/", IndexHandler),
        (r"/(.*)", tornado.web.StaticFileHandler, {'path': './files', 'default_filename': 'index.html'}),
    ])
    server = tornado.httpserver.HTTPServer(app, max_buffer_size=500000)  # 500Ko
    server.listen(PORT)
    print(f"listening on port {PORT}")
    tornado.ioloop.IOLoop.current().start()

