#!/bin/bash

# Define the input and output files
input_file="parametric_expressivity.md"
output_file="parametric_expressivity.pdf"

# Check if the input file exists
if [ -f "$input_file" ]; then
    echo "Converting $input_file to PDF..."
    pandoc "$input_file" -o "$output_file" --pdf-engine=pdflatex
    echo "Conversion complete: $output_file created."
else
    echo "Error: $input_file not found."
    exit 1
fi
