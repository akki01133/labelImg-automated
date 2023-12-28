from ultralytics import YOLO
import fer
import cv2
import os


class FERTool:
    def __init__(self, yolo_model_path):
        self.yolo_model_path = yolo_model_path
        self.yolo_model = YOLO(self.yolo_model_path)
        self.fer_model = fer.FER(mtcnn=True)

    def predict(self, image_path, img_size, classification_type=None):
        # to explicitly save the yolo results to a file, use save=True
        results = self.yolo_model.predict(image_path)
        res = results[0]
        # variable to store the face rectangles to pass to fer model
        lines = []

        if classification_type == 'emotion':
            lines = self.classify_emotions(
                image_path, res.boxes.xyxy, img_size)
            return lines
        else:
            lines = []
            for i, xywh in enumerate(res.boxes.xywh):
                label = int(res.boxes.cls[i].item())
                if (label != 0):
                    continue
                x, y, w, h = [int(tensor.item()) for tensor in xywh]
                # show the image and the bounding box predictions using opencv
                x_center = float(x)/img_size[1]
                y_center = float(y)/img_size[0]
                w = float(w)/img_size[1]
                h = float(h)/img_size[0]
                x_min = max(float(x_center) - float(w) / 2, 0)
                x_max = min(float(x_center) + float(w) / 2, 1)
                y_min = max(float(y_center) - float(h) / 2, 0)
                y_max = min(float(y_center) + float(h) / 2, 1)

                x_min = round(img_size[1] * x_min)
                x_max = round(img_size[1] * x_max)
                y_min = round(img_size[0] * y_min)
                y_max = round(img_size[0] * y_max)

                points = [(x_min, y_min), (x_max, y_min),
                          (x_max, y_max), (x_min, y_max)]
                lines.append((res.names[label], points))
            cv2.waitKey(0)
            return lines

    def classify_emotions(self, image_path, faces_xyxy, img_size):
        img = cv2.imread(image_path)
        results = self.fer_model.detect_emotions(img)
        lines = []
        for i, r in enumerate(results):
            # single line format = image_name, x,y,w,h,emotion
            line = ''
            # get the emotion with highest probability , format = {'angry': 0.1, 'disgust': 0.0, 'fear': 0.0, 'happy': 0.0, 'sad': 0.0, 'surprise': 0.0, 'neutral': 0.9}
            emotions = r['emotions']
            label = max(emotions, key=emotions.get)
            # get the box coordinates, format = x,y,w,h
            x1, y1, w, h = r['box']
            x1 = float(x1)/img_size[1]
            y1 = float(y1)/img_size[0]
            w = float(w)/img_size[1]
            h = float(h)/img_size[0]
            x_center = x1+(w/2.0)
            y_center = y1+(h/2.0)
            x_min = max(float(x_center) - float(w) / 2, 0)
            x_max = min(float(x_center) + float(w) / 2, 1)
            y_min = max(float(y_center) - float(h) / 2, 0)
            y_max = min(float(y_center) + float(h) / 2, 1)

            x_min = round(img_size[1] * x_min)
            x_max = round(img_size[1] * x_max)
            y_min = round(img_size[0] * y_min)
            y_max = round(img_size[0] * y_max)

            points = [(x_min, y_min), (x_max, y_min),
                      (x_max, y_max), (x_min, y_max)]
            lines.append((label, points))
