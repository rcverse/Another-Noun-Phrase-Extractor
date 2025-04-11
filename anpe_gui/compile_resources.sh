#!/bin/bash
echo "Compiling Qt resources..."
pyrcc6 -o resources.rcc resources.qrc
if [ $? -ne 0 ]; then
    echo "Error compiling resources! Make sure PyQt6 is installed and pyrcc6 is in your PATH."
    exit 1
fi
echo "Resources compiled successfully to resources.rcc" 