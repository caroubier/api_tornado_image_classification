import tornado.web
import tornado.ioloop
from call_model import *
import io
from PIL import Image
import uuid

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

        img = Image.open(img)
        processed_img = old_processing(img)
        result = model_predict(model, processed_img)
        self.write(f"{result}")

class IndexHandler(tornado.web.RequestHandler):
    def get(self):
        self.render('index.html')


class VsdHandler(tornado.web.RequestHandler):
    def post(self):
        valid_ext = ['JPG', 'JPEG', 'PNG']

        try:
            file1 = self.request.files["fileImage"][0]
            value = self.get_argument('model_class')
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

        fname = str(uuid.uuid4())+"."+file_ext.lower()
        image_folder = os.path.join(SAMPLES_DATASET, str(value), fname)

        try:
            img.save(image_folder)
        except:
            self.write(f"Unable to keep file!")
            self.finish()

        self.write(f"thank you for your return !")

if __name__ == "__main__":
    ROOT_PATH = os.getcwd()
    IMAGE_PATH = "image"
    SAMPLES_DATASET = "samples_dataset"
    MODEL_FOLDER = "models"

    if not os.path.exists(IMAGE_PATH):
        os.makedirs(IMAGE_PATH)

    if not os.path.exists(SAMPLES_DATASET):
        os.makedirs(SAMPLES_DATASET)

    for i in range(5):
        a = i + 1
        a = str(a)
        if not os.path.exists((os.path.join(SAMPLES_DATASET, a))):
            os.mkdir(os.path.join(SAMPLES_DATASET, a))

    model_path = os.path.join(ROOT_PATH,MODEL_FOLDER)
    model_name = "best_CNN_Conv32_MaxPool2_Conv64_MaxPool2_Conv32_MaxPool2_Dense64relu_Dropout05_Dense32Relu_Dropout05_Dense6Relu.h5"
    model = load_model(os.path.join(model_path, model_name))

    app = tornado.web.Application([
        (r"/image", ImgClassifierHandler),
        (r"/uvsd", VsdHandler), # user  validated sample dataset / response validated by user
        (r"/", IndexHandler)
    ])
    server = tornado.httpserver.HTTPServer(app, max_buffer_size=200000)  # 2Mo
    server.listen(5050)
    print("listening on port 5050")
    tornado.ioloop.IOLoop.current().start()

