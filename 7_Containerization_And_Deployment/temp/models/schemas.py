
from pydantic import BaseModel



class image_predRequestModel(BaseModel):
    image: str
    class_name : str