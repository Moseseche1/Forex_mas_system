from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import os

router = APIRouter()

# Setup templates
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))

@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@router.get("/", response_class=HTMLResponse)
async def root_dashboard(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
  
