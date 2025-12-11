import warnings
warnings.filterwarnings("ignore")

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn
from Models.MultiModal import BasicMultimodalModel
from Models.Encoder import SimpleEncoder1D

app = FastAPI()
class InputData(BaseModel):
    data: list[float]

class OutputData(BaseModel):
    """
    Result of the model
    dict type
    {
        "질문 이름": str,
        "질문 답변": int,
    }
    """
    result: dict[str, int]

if __name__ == "__main__":

    required_mods = ['mLight','mACStatus', 'mActivity', 'mBle', 'mGps', 'wHr',
                     'mScreenStatus', 'mUsageStats', 'mWifi', 'wLight', 'wPedo']
    feature_dims = {
        "mLight": 2,
        "mBle": 12,
        "mACStatus": 2,
        "wPedo": 14,
        "mGps": 14,
        "mUsageStats": 6,
        "mActivity": 2,
        "wHr": 8,
        "wLight": 2,
        "mWifi": 8,
        "mScreenStatus": 2,
    }
    DROPOUT_RATIO = 0.1
    LR = 1e-3
    WEIGHT_DECAY = 1e-5
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    INPUT_DIM = {k:288 for k in required_mods}  # 5min 간격, 24*60/5 = 288
    LATENT_DIM = 32
    FUSION_DIM = 128
    BS = 32
    encoder_dict = {}
    modality_names = required_mods

    for modality_name in modality_names:
        encoder_dict[modality_name] = SimpleEncoder1D(
            in_channels=feature_dims[modality_name],
            latent_dim=LATENT_DIM,
            out_dim=FUSION_DIM,
        )

    model = BasicMultimodalModel(
        modality_names=modality_names,
        encoder_dict=encoder_dict,
        fusion_dim=FUSION_DIM,
        dropout_ratio=DROPOUT_RATIO,
        task_clsss_dict={"Q1":2, "Q2":2, "Q3":2, "S1":3, "S2":2, "S3":2}
    ).to(DEVICE)

    model.load_state_dict(torch.load(os.path.join("Weights","ETRI2024","best_model.pth"), map_location=DEVICE))
    model.eval()

    @app.post("/predict")
    def predict(input_data: InputData):
        with torch.no_grad():
            input_data = input_data.data
            input_data = torch.tensor(input_data).to(DEVICE)
            output = model.predict(input_data)
            return OutputData(result=output)