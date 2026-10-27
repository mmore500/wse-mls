#!/bin/bash

set -e

echo "ruff check..."
ruff check --ignore=E501,E402 $@ .

echo "nbqa ruff check..."
python3 -m nbqa ruff --ignore=E501,E402 $@ .
