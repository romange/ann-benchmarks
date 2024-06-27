#!/bin/bash
set -ex

cd /tmp
ARCH=`uname -m`
BASE=https://github.com/dragonflydb/dragonfly/releases/latest/download

if [ "$ARCH" == "x86_64" ]; then
   PACKAGE_URL="$BASE/dragonfly-x86_64.tar.gz"
elif [ "$ARCH" == "aarch64" ]; then
   PACKAGE_URL="BASE/dragonfly-aarch64.tar.gz"
else
    echo "Unsupported architecture: $ARCH"
    exit 1
fi

curl -L -s $PACKAGE_URL -o /tmp/dragonfly.tar.gz
tar xvfz /tmp/dragonfly.tar.gz && mv /tmp/dragonfly-* /usr/local/bin/dragonfly
