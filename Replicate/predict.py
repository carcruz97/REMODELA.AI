from skimage import transform,io
from skimage.draw import rectangle_perimeter
from typing import Any
from io import BytesIO
import requests
import matplotlib.pyplot as plt
from cog import BasePredictor, Input

class API_Managment:
    def __init__(self,image_path):
        self.image_path=image_path
    def ultralytcis(self):
        url_model="https://api.ultralytics.com/v1/predict/XXXXXXX"
        api_key="XXXXXXX"
        headers = {"x-api-key": api_key}
        data_dict = {
                "size": 800,
                "confidence": 0.25,
                "iou": 0.45,
        }
        image_data = requests.get(self.image_path).content
        files = {"image": image_data}
        image_data_io = BytesIO(image_data)
        image = io.imread(image_data_io)
        response = requests.post(url_model , headers=headers, data=data_dict, files=files)
        if response.status_code == 200:
            raise requests.exceptions.HTTPError(500, "Model request failed")
        object_detection_results = response.json()['data']
        self.real_size_object = RealSizeObject(image, object_detection_results)
        return self.real_size_object
    
    def upload_to_imgbb(filename):
        with open(filename, 'rb') as image_file:
            response = requests.post(
                "https://api.imgbb.com/1/upload",
                params={
                    "expiration": 600,
                    "key": "XXXXXXX"  # Replace with your actual API key
                },
                files={"image": image_file}
            )

        if response.status_code == 200:
            imgbb_url = response.json()['data']['url']
            return f"Image processed and uploaded to imgBB. URL: {imgbb_url}"
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
      resized_image = transform.resize(self.indoor, (800, 600))  # dimensionalidad
      plt.imshow(resized_image)
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
          'perimeter': round(self.scale * (self.scale * abs(coords['x2'] - coords['x1']) +self.scale * abs(coords['y2'] - coords['y1'])),3)
          } for obj, coords in self.output_predict.items()}

      return self.real_size






class Predictor(BasePredictor):
    
    def setup(self):
        self.real_size_object = API_Managment(self.image_path).ultralytcis()

    def predict(self,image_path: str=Input(description="Paste the url of the image"),option_input: int=Input(description="Turn 1 or 0"),ref_object: str=Input(description="What object did you choose as a reference?"),ref_measure:str=Input(description="Enter the measure"),ref_pattern: float = Input(description="Is it height or width?"))-> Any:
        self.image_path=image_path
        self.ref_object=ref_object
        self.ref_measure=ref_measure
        self.ref_pattern=ref_pattern
        self.option_input=option_input
        image_with_labels = self.real_size_object.label_predictions()
        if self.option_input==1:
            return image_with_labels
        else:

            scaled_objects =self.real_size_object.scaling_object(self.ref_pattern, self.ref_object, self.ref_measure)
            return scaled_objects


