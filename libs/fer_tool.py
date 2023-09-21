from ultralytics import YOLO
import fer 
import cv2
import os

class FERTool:
    def __init__(self, yolo_model_path):
        self.yolo_model_path = yolo_model_path
        self.yolo_model = YOLO(self.yolo_model_path)
        self.fer_model = fer.FER(mtcnn=True)
        
    def predict(self, image_path, img_size):
        # to explicitly save the yolo results to a file, use save=True
        results = self.yolo_model.predict(image_path)
        res = results[0]
        # variable to store the face rectangles to pass to fer model
        face_rectangles = []
        for xyxy in res.boxes.xyxy:
            x1,y1, x2, y2 = [int(tensor.item()) for tensor in xyxy]
            face_rectangles.append((x1,y1,x2-x1,y2-y1))
        img = cv2.imread(image_path)
        results = self.fer_model.detect_emotions(img, face_rectangles=face_rectangles)
        lines = []
        for i,r in enumerate(results):
            # single line format = image_name, x,y,w,h,emotion
            line = ''
            # get the emotion with highest probability , format = {'angry': 0.1, 'disgust': 0.0, 'fear': 0.0, 'happy': 0.0, 'sad': 0.0, 'surprise': 0.0, 'neutral': 0.9}
            emotions = r['emotions']
            emotion = max(emotions, key=emotions.get)
            # get the box coordinates, format = x,y,w,h
            x1,y1,w,h = r['box']
            x1 = float(x1)/img_size[1]; y1 = float(y1)/img_size[0]; w = float(w)/img_size[1]; h = float(h)/img_size[0]
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

            points = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
            lines.append((emotion, points))
        return lines    
    
    def save_prediction_results(self, image_name, results, output_dir):
        '''
            the results should be non empty.
            the output path should be non empty and it is directory path
        '''
        # change the image_name extension to txt
        image_name = os.path.splitext(image_name)[0] + ".txt"    
        # create the output file if it does not exist
        output_path = os.path.join(output_dir, image_name)
        if not os.path.exists(output_path):
            open(output_path, 'w').close()
        # open the output file in append mode
        with open(output_path, 'w') as f:
            # concatenate the lines in the results list with newline characters
            output_str = "\n".join(results)
            # write the concatenated string to the file
            f.write(output_str)

            




