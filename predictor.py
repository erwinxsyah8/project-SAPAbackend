from .inference import predict

def predict_ocean(texts):
    """
    Wrapper untuk backend main.py
    """
    return predict(texts)
