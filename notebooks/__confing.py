import os

def change_to_root_folder():
    # Get the absolute path of the current script file
    script_path = os.path.abspath(__file__)
    # Get the root folder of the project (assuming it is one level above the script file)
    root_folder = os.path.dirname(os.path.dirname(script_path))
    # Change to the root folder directory
    os.chdir(root_folder)
    # Verify the new current directory
    current_directory = os.getcwd()
    print("Current directory:", current_directory)