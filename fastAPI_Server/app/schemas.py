from pydantic import BaseModel, EmailStr
from datetime import datetime

class RegisterIn(BaseModel):
    email: EmailStr
    password: str
    display_name: str | None = None

class LoginIn(BaseModel):
    email: EmailStr
    password: str

class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"

class MemberOut(BaseModel):
    id: str
    email: str
    display_name: str | None
    role: str
    is_active: bool

class PhotoPresignIn(BaseModel):
    filename: str
    content_type: str | None = None
    img_x: float
    img_y: float
    geo_x: float
    geo_y: float
    rdid: str

class PhotoPresignOut(BaseModel):
    session_id: str
    upload_url: str
    object_key: str
    bucket: str
    expires_in: int

class PhotoOut(BaseModel):
    id: str
    object_key: str
    bucket: str
    original_filename: str | None
    content_type: str | None
    size_bytes: int | None
    created_at: datetime
    img_x: float
    img_y: float
    geo_x: float
    geo_y: float
    rdid: str

class InferenceResultOut(BaseModel):
    id: str
    photo_id: str
    status: str
    result_object_key: str | None = None
    result_json: dict | None = None
    error_message: str | None = None
    rdid: str | None = None
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    size_bytes: int | None = None

class InferenceCallbackIn(BaseModel):
    job_id: str
    status: str
    result_object_key: str | None = None
    result_json: dict | None = None
    error_message: str | None = None
    size_bytes: int | None = None
