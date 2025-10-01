import requests
import sys
import json
import base64
import io
from datetime import datetime
from PIL import Image
import numpy as np

class AutismMonitoringAPITester:
    def __init__(self, base_url="https://autism-monitor.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.tests_run = 0
        self.tests_passed = 0
        self.session_id = f"test_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def run_test(self, name, method, endpoint, expected_status, data=None, files=None):
        """Run a single API test"""
        url = f"{self.api_url}/{endpoint}" if not endpoint.startswith('http') else endpoint
        headers = {'Content-Type': 'application/json'} if not files else {}

        self.tests_run += 1
        print(f"\n🔍 Testing {name}...")
        print(f"   URL: {url}")
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers, timeout=30)
            elif method == 'POST':
                if files:
                    response = requests.post(url, files=files, data=data, timeout=60)
                else:
                    response = requests.post(url, json=data, headers=headers, timeout=60)

            success = response.status_code == expected_status
            if success:
                self.tests_passed += 1
                print(f"✅ Passed - Status: {response.status_code}")
                try:
                    response_data = response.json()
                    print(f"   Response keys: {list(response_data.keys()) if isinstance(response_data, dict) else 'Non-dict response'}")
                except:
                    print(f"   Response: {response.text[:100]}...")
            else:
                print(f"❌ Failed - Expected {expected_status}, got {response.status_code}")
                print(f"   Error: {response.text[:200]}...")

            return success, response.json() if response.headers.get('content-type', '').startswith('application/json') else response.text

        except Exception as e:
            print(f"❌ Failed - Error: {str(e)}")
            return False, {}

    def create_test_image(self):
        """Create a test image for frame analysis"""
        # Create a simple test image
        img = Image.new('RGB', (640, 480), color='blue')
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG')
        img_data = buffer.getvalue()
        return base64.b64encode(img_data).decode('utf-8')

    def create_test_video_file(self):
        """Create a minimal test video file"""
        # Create a simple test file (not a real video, but for upload testing)
        test_content = b"fake video content for testing"
        return ('test_video.mp4', test_content, 'video/mp4')

    def test_root_endpoint(self):
        """Test root API endpoint"""
        return self.run_test("Root API", "GET", "", 200)

    def test_activities_get(self):
        """Test getting activities"""
        return self.run_test("Get Activities", "GET", "activities", 200)

    def test_activities_post(self):
        """Test creating activity detection"""
        activity_data = {
            "activity_type": "sitting",
            "confidence": 0.85,
            "bounding_box": {"x": 100, "y": 100, "width": 200, "height": 300},
            "description": "Test activity detection",
            "video_source": "test",
            "session_id": self.session_id
        }
        return self.run_test("Create Activity", "POST", "activities", 200, data=activity_data)

    def test_analyze_frame(self):
        """Test frame analysis endpoint"""
        test_frame = self.create_test_image()
        frame_data = {
            "frame": test_frame,
            "session_id": self.session_id
        }
        return self.run_test("Analyze Frame", "POST", "analyze_frame", 200, data=frame_data)

    def test_upload_video(self):
        """Test video upload endpoint"""
        # Note: This will likely fail with our fake video, but tests the endpoint
        test_file = self.create_test_video_file()
        files = {'file': test_file}
        data = {'session_id': self.session_id}
        success, response = self.run_test("Upload Video", "POST", "upload_video", 200, data=data, files=files)
        # Accept 400 as valid since we're sending fake video data
        if not success:
            # Try again expecting 400 for invalid video
            self.tests_run -= 1  # Don't double count
            return self.run_test("Upload Video (expect 400)", "POST", "upload_video", 400, data=data, files=files)
        return success, response

    def test_analytics_summary(self):
        """Test analytics summary endpoint"""
        return self.run_test("Analytics Summary", "GET", "analytics/summary", 200)

    def test_analytics_timeline(self):
        """Test analytics timeline endpoint"""
        return self.run_test("Analytics Timeline", "GET", "analytics/timeline", 200)

    def test_labeling_get(self):
        """Test getting labeling data"""
        return self.run_test("Get Labeling Data", "GET", "labeling", 200)

    def test_labeling_post(self):
        """Test creating labeling data"""
        test_frame = self.create_test_image()
        labeling_data = {
            "video_frame": test_frame,
            "labels": [{"activity": "sitting", "confidence": 1.0}],
            "annotator_id": "test_user",
            "session_id": self.session_id
        }
        return self.run_test("Create Labeling Data", "POST", "labeling", 200, data=labeling_data)

    def test_alerts(self):
        """Test alerts endpoint"""
        return self.run_test("Get Alerts", "GET", "alerts", 200)

    def run_all_tests(self):
        """Run all API tests"""
        print("🚀 Starting Autism Monitoring API Tests")
        print(f"📍 Base URL: {self.base_url}")
        print(f"🔗 API URL: {self.api_url}")
        print(f"🆔 Session ID: {self.session_id}")
        
        # Test all endpoints
        tests = [
            self.test_root_endpoint,
            self.test_activities_get,
            self.test_activities_post,
            self.test_analyze_frame,
            self.test_upload_video,
            self.test_analytics_summary,
            self.test_analytics_timeline,
            self.test_labeling_get,
            self.test_labeling_post,
            self.test_alerts
        ]
        
        for test in tests:
            try:
                test()
            except Exception as e:
                print(f"❌ Test {test.__name__} failed with exception: {e}")
        
        # Print final results
        print(f"\n📊 Test Results:")
        print(f"   Tests Run: {self.tests_run}")
        print(f"   Tests Passed: {self.tests_passed}")
        print(f"   Success Rate: {(self.tests_passed/self.tests_run*100):.1f}%")
        
        return self.tests_passed, self.tests_run

def main():
    tester = AutismMonitoringAPITester()
    passed, total = tester.run_all_tests()
    
    # Return appropriate exit code
    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())