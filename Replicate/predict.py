from skimage.draw import rectangle_perimeter
from typing import Any
import requests
import matplotlib.pyplot as plt
from cog import BasePredictor, Input,File
from PIL import Image
from io import BytesIO
import numpy as np
class API_Managment:
    def __init__(self,image_path):
        self.image_path=image_path
    def ultralytcis(self):
        url_model="https://api.ultralytics.com/v1/predict/XXXXXXXXX"
        api_key="APY-KEY-XXXXXX"
        headers = {"x-api-key": api_key}
        data_dict = {
                "size": 800,
                "confidence": 0.25,
                "iou": 0.45,
        }


        image_data = Image.open(self.image_path)
        target_height = int((640 / float(image_data.size[0])) * image_data.size[1])
        resized_image = image_data.resize((640, target_height), Image.ANTIALIAS)

        image = np.array(resized_image)


        with BytesIO() as buffer:
            resized_image.save(buffer, format="JPEG")
            image_bytes = buffer.getvalue()
            response = requests.post(url_model, headers=headers, data=data_dict, files={"image": image_bytes})
        object_detection_results = response.json()['data']
        self.real_size_object = RealSizeObject(image, object_detection_results)
        return self.real_size_object
    
    def upload_to_imgbb(filename):
        with open(filename, 'rb') as image_file:
            response = requests.post(
                "https://api.imgbb.com/1/upload",
                params={
                    "expiration": 1200,
                    "key": "f55bea8fb8190d88688ab7b501f758c9"  # Replace with your actual API key
                },
                files={"image": image_file}
            )

        if response.status_code == 200:
            imgbb_url = response.json()['data']['url']
            return "A development for REMODELA.AI. \n" +"Copyright © All rights reserved byrushrafa \n"+"Contact: byrushrafa@hotmail.com\n"+"github: @carcruz97\n"+f"Image processed and uploaded to imgBB. URL: {imgbb_url}"
        else:
            return "Error uploading the image to imgBB."

class RealSizeObject:
    def __init__(self,image_data,raw_predict):
      self.indoor=image_data
      boxes_predict = [{k: v for k, v in item.items() if k not in ('class', 'confidence','name')} for item in raw_predict]
      self.output_predict= {f'obj {i+1}': item.pop('box') for i, item in enumerate(boxes_predict)}

    def label_predictions(self):

      for name, coord in self.output_predict.items():
        x1, y1,x2, y2= int(coord['x1']), int(coord['y1']), int(coord['x2']), int(coord['y2'])
        rr, cc = rectangle_perimeter(start=(y1, x1), end=(y2, x2), shape=self.indoor.shape, clip=True)
        self.indoor[rr, cc] = [255, 0, 0] #red color
        text_x, text_y = x1, y1- 10
        plt.text(text_x, text_y, name, color='black', fontsize=7, backgroundcolor='white')
      plt.imshow(self.indoor)
      plt.savefig('Real_Size_Objects_Indoor.png', bbox_inches='tight', pad_inches=0, format='png')
      plt.close()
      image_result=API_Managment.upload_to_imgbb('Real_Size_Objects_Indoor.png')
      return image_result
    

    def scaling_object(self,ref_pattern,ref_object,ref_measure):
      self.ref_pattern = ref_pattern
      self.ref_object=ref_object
      self.ref_measure=ref_measure

      self.ref_object=self.output_predict[self.ref_object] #objecto de referencia que envia el usuario
      self.pixel = abs(self.ref_object['y1'] - self.ref_object['y2']) if self.ref_measure.lower() in ["alto","height"] else abs(self.ref_object['x1'] - self.ref_object['x2']) #medida del objecto en pixeles
      self.scale=round(self.ref_pattern/self.pixel,3)

      self.real_size= {obj: {
          'width': round(self.scale*abs(coords['x2'] - coords['x1']),3),
          'height': round(self.scale*abs(coords['y2'] - coords['y1']),3),
          'area':round(self.scale*self.scale* abs(coords['x2'] - coords['x1']) * abs(coords['y2'] - coords['y1']),3),
          'perimeter': round(2* (self.scale * abs(coords['x2'] - coords['x1']) +self.scale * abs(coords['y2'] - coords['y1'])),3)
          } for obj, coords in self.output_predict.items()}

      return self.real_size

class Predictor(BasePredictor):
    def predict(self,image_path: File = Input(description="Image to enlarge"),select_option: int=Input(description="Upload Image write '1'. Calculate Size Objects write '0'"),ref_object: str=Input(description="What object did you choose as a reference?"),ref_measure:str=Input(description="Is it width or height?"),ref_pattern: float = Input(description="What is the size in centimetres?"))-> Any:
        self.image_path=image_path
        self.ref_object=ref_object
        self.ref_measure=ref_measure
        self.ref_pattern=ref_pattern
        self.select_option=select_option
        self.real_size_object = API_Managment(self.image_path).ultralytcis()
        image_with_labels = self.real_size_object.label_predictions()
        if self.select_option==1:
            return image_with_labels
        else:

            scaled_objects =self.real_size_object.scaling_object(self.ref_pattern, self.ref_object, self.ref_measure)
            return "Scaling of spaces in centimetres: ",scaled_objects

