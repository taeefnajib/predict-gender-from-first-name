from pydantic import BaseModel
from typing import List
import numpy as np

class Name(BaseModel):
    name: str