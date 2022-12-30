import os
current_dir = "/cache/train/checkpoint"
def useOSWalk(folder):
    num = 1
    for root,folder_names, file_names in os.walk(folder):
        print(root)
        print(folder_names)
        print(file_names)
        print(num)
        num+=1
        
if __name__ == "__main__":
    useOSWalk(current_dir)
