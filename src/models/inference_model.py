from models.base_model import BaseModel

class InferenceModel(BaseModel):
    
    def __init__(self):
        
        super(InferenceModel,self).__init__()
        self.init_parameters()