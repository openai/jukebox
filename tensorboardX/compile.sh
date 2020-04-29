#!/bin/bash

# Exit on error
# set -e

DESIRED_PROTO_VERSION="3.6.1"

# call protoc direclty, if version is not the desired one, download the desired vesrion.


if [ -f "protoc/bin/protoc" ]; then
  PROTOC_BIN="protoc/bin/protoc"
else
  PROTOC_BIN=`which protoc`
fi

echo "using" $PROTOC_BIN

CURRENT_PROTOC_VER=`${PROTOC_BIN} --version`
if [ -z ${PROTOC_BIN} ] || [[ "$CURRENT_PROTOC_VER" != "libprotoc "$DESIRED_PROTO_VERSION ]]; then
  # Download and use the latest version of protoc.
  if [ "$(uname)" == "Darwin" ]; then
    PROTOC_ZIP="protoc-"$DESIRED_PROTO_VERSION"-osx-x86_64.zip"
  else
    PROTOC_ZIP="protoc-"$DESIRED_PROTO_VERSION"-linux-x86_64.zip"
  fi
  WGET_BIN=`which wget`
  if [[ ! -z ${WGET_BIN} ]]; then
    ${WGET_BIN} https://github.com/protocolbuffers/protobuf/releases/download/v"$DESIRED_PROTO_VERSION"/${PROTOC_ZIP}
    rm -rf protoc
    python -c "import zipfile; zipfile.ZipFile('"${PROTOC_ZIP}"','r').extractall('protoc')"
    PROTOC_BIN=protoc/bin/protoc
    chmod +x ${PROTOC_BIN}
  fi
fi

# Regenerate
if [[ ! -z ${PROTOC_BIN} ]]; then
  # Delete all existing Python protobuf (*_pb2.py) output
  rm -rf tensorboardX/proto/*pb2*.py
  ${PROTOC_BIN} tensorboardX/proto/*.proto --python_out=.

  echo "Done generating tensorboardX/proto/*pb2*.py"
else
  echo "protoc not installed so can't regenerate tensorboardX/proto/*pb2*.py, using precompiled version."
fi

