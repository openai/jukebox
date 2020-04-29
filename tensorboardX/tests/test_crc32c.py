import unittest
from tensorboardX.crc32c import _crc32c, _crc32c_native, crc32c


class CRC32CTest(unittest.TestCase):
    def test_crc32c(self):
        data = b'abcd'
        assert crc32c(data) == 0x92c80a31

    def test_crc32c_python(self):
        data = b'abcd'
        assert _crc32c(data) == 0x92c80a31

    def test_crc32c_native(self):
        if _crc32c_native is None:
            return
        data = b'abcd'
        assert _crc32c_native(data) == 0x92c80a31
