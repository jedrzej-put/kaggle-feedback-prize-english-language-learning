#!/bin/bash

# Check if a path is provided
if [ $# -eq 0 ]; then
    echo "Please provide a file or directory path"
    exit 1
fi

# Path provided
PATH_ARG="$1"

# Function to process a Python file
process_python_file() {
    local file="$1"
    echo "Processing Python file: $file"
    black --line-length=119 "$file"
    autopep8 --in-place --aggressive --aggressive --max-line-length=119 "$file"
    isort --profile=black --line-length=119 "$file"
    flake8 --max-line-length=119 "$file"
}

# Function to process a Jupyter notebook
process_notebook() {
    local notebook="$1"
    echo "Processing Jupyter notebook: $notebook"
    nbqa black --line-length=119 "$notebook"
    nbqa autopep8 --in-place --aggressive --aggressive --max-line-length=119 "$notebook"
    nbqa isort --profile=black --line-length=119 "$notebook"
    nbqa flake8 --max-line-length=119 "$notebook"
}

# Function to process files in a directory
process_directory() {
    local dir_path="$1"
    echo "Processing directory: $dir_path"
    
    # Process Python files
    find "$dir_path" -name "*.py" -type f | while read -r file; do
        process_python_file "$file"
    done

    # Process Jupyter notebooks
    find "$dir_path" -name "*.ipynb" -type f | while read -r notebook; do
        process_notebook "$notebook"
    done
}

# Main execution
echo "Starting code formatting and linting..."

if [ -f "$PATH_ARG" ]; then
    # It's a file
    if [[ "$PATH_ARG" == *.py ]]; then
        process_python_file "$PATH_ARG"
    elif [[ "$PATH_ARG" == *.ipynb ]]; then
        process_notebook "$PATH_ARG"
    else
        echo "Unsupported file type. Please provide a .py or .ipynb file."
        exit 1
    fi
elif [ -d "$PATH_ARG" ]; then
    # It's a directory
    process_directory "$PATH_ARG"
else
    echo "The provided path is neither a file nor a directory."
    exit 1
fi

echo "Finished processing."