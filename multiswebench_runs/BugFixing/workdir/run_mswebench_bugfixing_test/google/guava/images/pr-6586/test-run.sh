#!/bin/bash
set -e
cd /home/guava
git apply --whitespace=nowarn /home/test.patch
mvn clean test -DfailIfNoTests=false
