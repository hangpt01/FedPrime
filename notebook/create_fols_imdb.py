import os
import shutil

def create_folders(source_folder):
    # Create 'images' and 'labels' folders
    os.makedirs(os.path.join(source_folder, 'images'), exist_ok=True)
    os.makedirs(os.path.join(source_folder, 'labels'), exist_ok=True)

def move_files(source_folder, des_folder):
    for filename in os.listdir(source_folder):
        if filename.endswith('.jpeg'):
            shutil.move(os.path.join(source_folder, filename),
                        os.path.join(des_folder, 'images', filename))
        elif filename.endswith('.json'):
            shutil.move(os.path.join(source_folder, filename),
                        os.path.join(des_folder, 'labels', filename))
    shutil.move("benchmark/RAW_DATA/IMDB/mmimdb/split.json", "benchmark/RAW_DATA/IMDB/split.json")
    
def main():
    source_folder = 'benchmark/RAW_DATA/IMDB/mmimdb/dataset'
    des_folder = "benchmark/RAW_DATA/IMDB"
    create_folders(des_folder)
    move_files(source_folder, des_folder)

if __name__ == "__main__":
    main()
