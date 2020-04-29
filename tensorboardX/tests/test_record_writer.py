from tensorboardX import SummaryWriter
import unittest
from tensorboardX.record_writer import S3RecordWriter, make_valid_tf_name
import os
import boto3
from moto import mock_s3

os.environ.setdefault("AWS_ACCESS_KEY_ID", "foobar_key")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "foobar_secret")


class RecordWriterTest(unittest.TestCase):
    @mock_s3
    def test_record_writer_s3(self):
        client = boto3.client('s3', region_name='us-east-1')
        client.create_bucket(Bucket='this')
        writer = S3RecordWriter('s3://this/is/apen')
        bucket, path = writer.bucket_and_path()
        assert bucket == 'this'
        assert path == 'is/apen'
        writer.write(bytes(42))
        writer.flush()

    def test_make_valid_tf_name(self):
        newname = make_valid_tf_name('$ave/&sound')
        assert newname == '._ave/_sound'
