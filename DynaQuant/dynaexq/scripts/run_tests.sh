#!/bin/bash
# Run DynaExQ unit tests

set -e

cd "$(dirname "$0")/../.."

echo "Running DynaExQ unit tests..."
echo "=============================="
echo ""

# Run tests with Python unittest
python -m unittest discover -s dynaexq/tests -p "test_*.py" -v

echo ""
echo "=============================="
echo "All tests passed!"

