"""
URL configuration for ai_project project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include  # Make sure include is imported
from chatbot.api import api  # Import the chatbot api object
from aimicroservice.face_recognition_api import facials_api  # Import the facial recognition api
from aimicroservice.document_ocr_api import ocr_api  # Import the document OCR api
from django.http import HttpResponse  # Added import for root view


# Health check or root view
def root(request):
    return HttpResponse("AI Microservice - RAG Chatbot, Facial Recognition & Document OCR Service Running")


urlpatterns = [
    path("", root, name="root"),  # Root path
    path('admin/', admin.site.urls),
    path("api/", api.urls),  # Chatbot API routes (/api/chat, /api/transactions, etc.)
    path("api/facials/", facials_api.urls),  # Facial recognition API routes (/api/facials/register, /api/facials/verify, etc.)
    path("api/ocr/", ocr_api.urls),  # Document OCR API routes (/api/ocr/process, /api/ocr/verify, etc.)
]
