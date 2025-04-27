import os

# Set search path to your Python installation directory
search_root = "C:/Users/17818/AppData/Local/Programs/Python/Python39/"

# Walk through directories to find TFLite files
for root, dirs, files in os.walk(search_root):
    for file in files:
        if file.endswith(".tflite"):
            print(f"Found model: {os.path.join(root, file)}")
