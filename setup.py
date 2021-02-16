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
        os.system(f"cp -r {os.path.join('demo','scripts')} {args.path}")
    # check temp folder
    temp_folder_path = os.path.join(args.path,"temp")
    if not os.path.exists(temp_folder_path):
        os.makedirs(temp_folder_path)
    # check model.py
    model_file_path = os.path.join(args.path,"models.py")
    if not os.path.exists(model_file_path):
        os.system(f"cp {os.path.join('demo','models.py')} {model_file_path}")
    # check train.py
    train_file_path = os.path.join(args.path,"train.py")
    if not os.path.exists(train_file_path):
        os.system(f"cp {os.path.join('demo','train.py')} {train_file_path}")
    # check utils.py
    utils_file_path = os.path.join(args.path,"utils.py")
    if not os.path.exists(utils_file_path):
        os.system(f"cp {os.path.join('demo','utils.py')} {utils_file_path}")

    print(f"Directory has been developed in {args.path}")
    print(os.system(f"tree {args.path}"))
