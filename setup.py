import os
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, default='.',help="The path will be developed Ming standard directory")
    args = parser.parse_args()
    # check root folder
    if not os.path.exists(args.path):
        raise Exception(f"Path not exist {args.path}.")
    # check data folder
    data_folder_path = os.path.join(args.path,"data")
    if not os.path.exists(data_folder_path):
        os.makedirs(data_folder_path)
    # check logs folder
    logs_folder_path = os.path.join(args.path,"logs")
    if not os.path.exists(logs_folder_path):
        os.makedirs(logs_folder_path)
    # check scripts folder
    scripts_folder_path = os.path.join(args.path,"scripts")
    if not os.path.exists(scripts_folder_path):
        os.makedirs(scripts_folder_path)
    # check temp folder
    temp_folder_path = os.path.join(args.path,"temp")
    if not os.path.exists(temp_folder_path):
        os.makedirs(temp_folder_path)
    # check model.py
    model_file_path = os.path.join(args.path,"model.py")
    if not os.path.exists(model_file_path):
        os.system(f"cp model.py {model_file_path}")
    # check train.py
    train_file_path = os.path.join(args.path,"train.py")
    if not os.path.exists(train_file_path):
        os.system(f"cp train.py {train_file_path}")
    # check utils.py
    utils_file_path = os.path.join(args.path,"utils.py")
    if not os.path.exists(utils_file_path):
        os.system(f"cp untils.py {utils_file_path}")
