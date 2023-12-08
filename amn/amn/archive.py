import os
import pathlib 
import zipfile
from pathlib import Path


def unzip_folders(main_directory):
    """Extract all the files from all the subdirectory of 
    the given directory."""
    sub_dir = os.listdir(main_directory)
    for d in sub_dir:
        if zipfile.is_zipfile(main_directory + d):
            unzip_folder(main_directory, d)


def unzip_folder(path_directory, folder_name):
    """Extract all files from a given folder of the given
    directory into this directory."""
    file = path_directory + folder_name
    print("Extracting",file)
    with zipfile.ZipFile(file, 'r') as zip_ref:
        zip_ref.extractall(path_directory)


def zip_models(model_directory):
    """Transform every file with ".keras" suffix in the given model
    directory into a compressed archive."""
    for file_name in os.listdir(model_directory):
        if pathlib.Path(model_directory + file_name).suffix == ".keras":
            zip_file(model_directory, file_name)


def zip_file(path_directory, file_name):
    """Transform a given file into a compressed archive in the same given
    directory."""
    file = path_directory + file_name
    archive_path = path_directory + Path(file_name).stem + ".zip"
    print("Create archive",archive_path)
    zf = zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED)
    zf.write(file, file_name)