#!/bin/bash

PROGLOC="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
CURLOC=`pwd`

cd $PROGLOC

rm -rf isotropic/evaluate/outputs
rm -rf isotropic/learn/outputs
rm -rf isotropic/visualize/outputs
rm -rf isotropic/evaluate/tmp
rm -rf isotropic/learn/models

rm -rf windy/evaluate/outputs
rm -rf windy/learn/outputs
rm -rf windy/visualize/outputs
rm -rf windy/evaluate/tmp
rm -rf windy/learn/models

rm -rf __pycache__
rm -rf */__pycache__
rm -rf */*/__pycache__
rm -rf */*/*/__pycache__
rm -rf .idea
rm -rf */.idea
rm -rf */*/.idea
rm -rf */*/*/.idea

cd $CURLOC
