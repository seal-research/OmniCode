#!/bin/bash
set -e
cd /home/guava
mvn clean test -DfailIfNoTests=false
