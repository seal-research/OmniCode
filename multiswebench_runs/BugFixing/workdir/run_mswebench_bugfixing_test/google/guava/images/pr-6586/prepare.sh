#!/bin/bash
set -e
cd /home/guava
git reset --hard
bash /home/check_git_changes.sh
git checkout 01dcc2e6104e9bd0392cb19029edf2c581425b67
bash /home/check_git_changes.sh

mvn clean test -DfailIfNoTests=false || true
