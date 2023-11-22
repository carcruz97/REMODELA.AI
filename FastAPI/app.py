from ScalingModel.FirstUseCase  import RealSizeObject
from skimage import io 
from io import BytesIO
from fastapi import FastAPI,HTTPException
from pydantic import BaseModel
import requests

app=FastAPI()

class ImageProcessingRequest(BaseModel):
    api_key: str
    image_path: str
    url_model: str
    confidence: float = 0.25
    iou: float = 0.45


class ScaleObject(BaseModel):
    ref_pattern: float
    ref_object: str
    ref_measure: str

async def process_image(data:ImageProcessingRequest):
    if not data.api_key:
        raise HTTPException(status_code=400, detail="API key is required")
    if not data.image_path:
        raise HTTPException(status_code=400, detail="Image file is required")
    image_data = requests.get(data.image_path).content
    files = {"image": image_data}
    image_data_io = BytesIO(image_data)
    image = io.imread(image_data_io)

    headers = {"x-api-key": data.api_key}
    data_dict = {
        "size": 800,
        "confidence": data.confidence,
        "iou": data.iou,
    }

    response = requests.post(data.url_model, headers=headers, data=data_dict, files=files)
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Model request failed")

    object_detection_results = response.json()['data']
    real_size_object = RealSizeObject(image, object_detection_results)
    return real_size_object
    
@app.get("/")
def read_root():
    return {"byrushrafa: A development for REMODELA.AI"}
    
@app.post("/process-image/")
async def label_object_image(data: ImageProcessingRequest):
    real_size_object = await  process_image(data)
    objects = real_size_object.label_predictions()
    return objects

@app.post("/scale/", response_model=dict)
async def scale_image(data: ImageProcessingRequest, scale_data: ScaleObject):
    real_size_object = await  process_image(data)

    if not scale_data.ref_pattern:
        raise HTTPException(status_code=400, detail="Reference pattern is required")

    if not scale_data.ref_object:
        raise HTTPException(status_code=400, detail="Reference object is required")

    if not scale_data.ref_measure:
        raise HTTPException(status_code=400, detail="Reference measure is required")
    
    scaled_objects =real_size_object.scaling_object(
        scale_data.ref_pattern, scale_data.ref_object, scale_data.ref_measure)

    return scaled_objects
