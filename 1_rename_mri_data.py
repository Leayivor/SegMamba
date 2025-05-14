import os 
import argparse 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="")
    return parser.parse_args()

args = parse_args()
data_dir = args.data_dir

all_cases = os.listdir(data_dir)

for case_name in all_cases:
    case_dir = os.path.join(data_dir, case_name)
    if not os.path.isdir(case_dir):
        continue

    for data_name in os.listdir(case_dir):
        if not data_name.startswith(case_name):
            continue    

        parts = data_name.split(f"{case_name}_")
        if len(parts) != 2:
            continue  

        new_name = parts[1]
        old_path = os.path.join(case_dir, data_name)
        new_path = os.path.join(case_dir, new_name)

        os.rename(old_path, new_path)
        print(f"{new_path} : Renaming Successful")

