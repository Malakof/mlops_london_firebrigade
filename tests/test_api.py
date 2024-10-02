import requests
import unittest

class TestAPI(unittest.TestCase):
    base_url = "http://127.0.0.1:8000"  # URL where the test server is running

    def test_health_endpoint(self):
        """Test the health check endpoint for a 200 response code."""
        response = requests.get(f"{self.base_url}/health")
        self.assertEqual(response.status_code, 200)

    def test_predict_endpoint(self):
        """Test the predict endpoint  """
        response = requests.get(f"{self.base_url}/predict?distance=1.3&station=Acton")
        self.assertEqual(response.status_code, 200)  # Assuming 401 is for unauthorized access

    def test_data_preprocessing_endpoint_authentification(self):
        """Test the data preprocessing endpoint authentification."""
        response = requests.get(f"{self.base_url}/process_data")
        self.assertEqual(response.status_code, 401)

if __name__ == '__main__':
    unittest.main()