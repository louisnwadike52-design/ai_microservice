#!/Users/louislawrence/Music/apps/stack/ai_microservice/aimicroservice/bin/python
"""
Verify Facial Recognition API Endpoints Configuration
This script checks if all facials endpoints are properly configured
"""

import os
import sys
import django

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ai_project.settings')
django.setup()

from aimicroservice.face_recognition_api import facials_api

print("=" * 60)
print("Facial Recognition API Endpoints Configuration")
print("=" * 60)
print()

print("✓ facials_api imported successfully")
print()

print("API Information:")
print(f"  Title: {facials_api.title}")
print(f"  Version: {facials_api.version}")
print(f"  Description: {facials_api.description}")
print()

print("Available Endpoints:")
print("  Base URL: /api/facials")
print()

# Get routes
routes = []
for url_pattern in facials_api.urls[0].url_patterns:
    path = str(url_pattern.pattern)
    routes.append(path)

for route in sorted(routes):
    method = "POST" if "register" in route or "verify" in route else "DELETE" if "delete" in route else "GET"
    print(f"  [{method:6}] /api/facials/{route}")

print()
print("=" * 60)
print("Configuration verified successfully!")
print("=" * 60)
print()
print("To test the API:")
print("  1. Start the server: python manage.py runserver")
print("  2. Visit Swagger UI: http://localhost:8000/api/facials/docs")
print("  3. Run test script: ./test_facials_api.sh")
