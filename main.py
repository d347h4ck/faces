from fastapi import FastAPI
from pydantic import BaseModel

from typing import Dict, Any
import numpy as np

import aiohttp
import asyncio

import base64
import cv2

import json

from deepface import DeepFace
from retinaface import RetinaFace
from enum import IntEnum


app = FastAPI()


class ImageType(IntEnum):
    EXTERNAL_LINK_JPEG = 0
    BASE64_JPEG = 1


class ImageURL(BaseModel):
    data: str
    image_type: ImageType

    class Config:  
        use_enum_values = True


@app.post("/detect")
async def detect_faces(url: ImageURL) -> Dict[str, Any]:
    itype = url.image_type
    match itype:
        case ImageType.EXTERNAL_LINK_JPEG:
            async with aiohttp.ClientSession() as session:
                async with session.get(url.data) as resp:
                    image = await resp.read()
                    np_pic = np.frombuffer(image, dtype=np.uint8)
                    img = cv2.imdecode(np_pic, flags=1)
        case ImageType.BASE64_JPEG:
            image = base64.b64decode(url.data)
            np_pic = np.frombuffer(image, dtype=np.uint8)
            img = cv2.imdecode(np_pic, flags=1)
        case _:
            return {"faces": [], "status_code": 3, "status_message": "Wrong image type"}    
    try:
        resp = RetinaFace.detect_faces(img)
    except ValueError:
        return {"faces": [], "status_code": 1, "status_message": "Image could not be load"}
    if (type(resp) != dict):
        return {"faces": [], "status_code": 2, "status_message": "Faces did not found"}
    
    res = {}
    faces = []
    for key in resp.keys():
        identity = resp[key]

        facial_area = np.array(identity['facial_area']).tolist()
        face = img[facial_area[1]: facial_area[3], facial_area[0]:facial_area[2]]
        repr_vec = np.array(DeepFace.represent(img_path = face, model_name = 'VGG-Face', enforce_detection=False, detector_backend = 'retinaface'))
        
        r = {"rectangle": {'left': facial_area[0],
                            'right': facial_area[2],
                            'top': facial_area[1],
                            'bottom': facial_area[3]
                          }, 
             "representation_vector": repr_vec.tolist()
            }
        faces.append(r)
    
    res['faces'] = faces
    res['status_code'] = 0
    res['status_message'] = ""
    return res
