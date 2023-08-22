import unittest
import camera

class TestCamera(unittest.TestCase):
    def test_getitem(self):
        self.assertIsInstance(camera.__getitem__(self, 0), camera.Camera)

    def test_getgroup(self):
        pass

    def test_getnclosestcameras(self):
        pass

#if __name__ == '__main__':
#    unittest.main()