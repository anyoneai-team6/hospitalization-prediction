import os
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import urllib.request
import zipfile
import platform
import pandas as pd

__X=['paweight','pagender','pachair','pabath','pacholst', 'parafaany', 'pacancre', 'paeat', 'paswell', 'paosleep', 'padiabe', 'paagey', 'paheight', 'padoctor1y', 'parjudg', 'pawheeze', 'paarthre', 'pahrtatte', 'parxstrok', 'padrinkb', 'papaina', 'pameds', 'pafallinj', 'pasmokev', 'padadage', 'pamomage', 'paclims', 'paglasses', 'pahearaid', 'pahipe_m', 'paoalchl', 'pastroke']

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


def execute_notebook(path):
    with open(path, 'r', encoding='utf-8') as file:
        content = file.read()
        nb = nbformat.reads(content, nbformat.NO_CONVERT)
    
    executor = ExecutePreprocessor(timeout=600)
    executor.preprocess(nb, {'metadata': {'path': '.'}})
    
    with open(path, 'w', encoding='utf-8') as file:
        nbformat.write(nb, file)


def create_folders():
    current_dir = os.getcwd()
    data_dir = os.path.join(current_dir, 'data')
    custom_dir = os.path.join(data_dir, 'custom')
    original_dir = os.path.join(data_dir, 'original')
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print("Created 'data' folder.")
    
    if not os.path.exists(original_dir):
        os.makedirs(original_dir)
        print("Created 'original' folder within 'data'.")
    
    if not os.path.exists(custom_dir):
        os.makedirs(custom_dir)
        print("Created 'custom' folder within 'data'.")


def download_zip():
    url = "https://www.mhasweb.org/resources/DATA/HarmonizedData/H_MHAS/Version_C/SAS/H_MHAS_c.sas.zip"
    output_dir = os.path.join("data", "original")
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    # Get the filename from the URL
    filename = os.path.basename(url)
    # Build the path to save the downloaded ZIP file
    zip_path = os.path.join(output_dir, filename)
    # Check if the ZIP file already exists
    if not os.path.exists(zip_path):
        try:
            # Download the ZIP file
            urllib.request.urlretrieve(url, zip_path)
            print("ZIP file downloaded successfully.")
        except Exception as e:
            print("Error occurred while downloading the ZIP file:", e)
            return  
    # Check if the ZIP file is already extracted
    extracted_files = [f for f in os.listdir(output_dir) if f != filename]
    if not extracted_files:
        try:
            # Extract the contents of the ZIP file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(output_dir)
            print("ZIP file extracted successfully.")
        except Exception as e:
            print("Error occurred while extracting the ZIP file:", e)
            # Clean up the extracted files
            extracted_files = [f for f in os.listdir(output_dir) if f != filename]
            for file in extracted_files:
                file_path = os.path.join(output_dir, file)
                os.remove(file_path)
    else:
        print("ZIP file already extracted.")


def get_data_frame():
    system = platform.system()
    
    if system == "Windows" or system == "Linux":
        data_path = r'data/original/H_MHAS_c.sas7bdat'
    elif system == "Darwin":  # macOS
        data_path = r'data/original/__MACOSX/._H_MHAS_c.sas7bdat'
    else:
        print("Unsupported operating system.")
        return None
    
    try:
        df = pd.read_sas(data_path)
        return df
    except Exception as e:
        print("Error occurred while reading the SAS file:", e)
        return None