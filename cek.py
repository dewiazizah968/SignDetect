import os

root_folder = 'D:\\PYTHON PROJECT\\Data Science UNESA\\Semester 4\\PCD\\Data Biner'

subfolders = [f for f in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, f))]

print(f"Total folder: {len(subfolders)}\n")

for folder in subfolders:
    folder_path = os.path.join(root_folder, folder)
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    print(f"Folder '{folder}' berisi {len(files)} file")