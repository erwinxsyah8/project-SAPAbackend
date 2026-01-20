from pydantic import BaseModel

class TextRequest(BaseModel):
    text: str

class OceanResponse(BaseModel):
    O: float
    C: float
    E: float
    A: float
    N: float

    O_round: int
    C_round: int
    E_round: int
    A_round: int
    N_round: int
