import tornado.web
import tornado.ioloop
from call_model import *
import io
from PIL import Image


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


if __name__ == "__main__":

    ROOT_PATH = os.getcwd()
    model_path = ROOT_PATH+"\\models\\"
    model_name = "best_CNN_Conv32_MaxPool2_Conv64_MaxPool2_Conv32_MaxPool2_Dense64relu_Dropout05_Dense32Relu_Dropout05_Dense6Relu.h5"
    model = load_model(model_path+model_name)

    app = tornado.web.Application([
        (r"/image", ImgClassifierHandler),
        # (r"/image", ImgClassifierHandler),
        (r"/", IndexHandler)
    ])
    server = tornado.httpserver.HTTPServer(app, max_buffer_size=200000)  # 10G
    server.listen(8888)
    print("listening on port 8888")
    tornado.ioloop.IOLoop.current().start()
