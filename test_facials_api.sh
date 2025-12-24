#!/bin/bash

# Test script for Facial Recognition API endpoints
# Make sure the AI microservice is running before executing this script

BASE_URL="http://localhost:8000/api/facials"

echo "=========================================="
echo "Testing Facial Recognition API Endpoints"
echo "=========================================="
echo ""

# Test 1: Health Check
echo "1. Testing Health Check Endpoint"
echo "GET ${BASE_URL}/health"
curl -X GET "${BASE_URL}/health" \
  -H "Content-Type: application/json" \
  -w "\nHTTP Status: %{http_code}\n" \
  -s | jq '.' || echo "Response received (jq not available)"
echo ""
echo "=========================================="
echo ""

# Test 2: Register Face (requires image file)
echo "2. Testing Register Face Endpoint"
echo "POST ${BASE_URL}/register"
echo "Note: This test requires a test image. Skipping actual test..."
echo "Example command:"
echo "curl -X POST '${BASE_URL}/register' \\"
echo "  -F 'user_id=test_user_123' \\"
echo "  -F 'image=@/path/to/face_image.jpg' \\"
echo "  -F 'allow_duplicates=false' \\"
echo "  -F 'duplicate_threshold=0.6'"
echo ""
echo "=========================================="
echo ""

# Test 3: Verify Face (requires image file)
echo "3. Testing Verify Face Endpoint"
echo "POST ${BASE_URL}/verify"
echo "Note: This test requires a test image. Skipping actual test..."
echo "Example command:"
echo "curl -X POST '${BASE_URL}/verify' \\"
echo "  -F 'user_id=test_user_123' \\"
echo "  -F 'image=@/path/to/face_image.jpg' \\"
echo "  -F 'threshold=0.5'"
echo ""
echo "=========================================="
echo ""

# Test 4: Get Face Count
echo "4. Testing Get Face Count Endpoint"
echo "GET ${BASE_URL}/count/test_user_123"
curl -X GET "${BASE_URL}/count/test_user_123" \
  -H "Content-Type: application/json" \
  -w "\nHTTP Status: %{http_code}\n" \
  -s | jq '.' || echo "Response received (jq not available)"
echo ""
echo "=========================================="
echo ""

# Test 5: Delete User Faces
echo "5. Testing Delete User Faces Endpoint"
echo "DELETE ${BASE_URL}/delete/test_user_123"
echo "Note: Uncomment below to actually delete test user faces"
echo "# curl -X DELETE '${BASE_URL}/delete/test_user_123' \\"
echo "#   -H 'Content-Type: application/json' \\"
echo "#   -w '\nHTTP Status: %{http_code}\n' \\"
echo "#   -s | jq '.'"
echo ""
echo "=========================================="
echo ""

echo "Test completed!"
echo ""
echo "API Documentation available at:"
echo "- Swagger UI: http://localhost:8000/api/facials/docs"
echo "- OpenAPI JSON: http://localhost:8000/api/facials/openapi.json"
