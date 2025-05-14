import json
import wandb
import pandas as pd
import argparse
import logging
import os
import yaml

# Access the run via W&B API

def download_files(run, id_experiment, output_dir, test=False):
    api = wandb.Api()
    run = api.run(f"{id_experiment}") #WATCH TO NOT OVERRIDE EXISTING NICKNAME
    logging.info(f"Downloading files from experiment: {run.name}")
    if test:
        output_root = os.path.join(output_dir, "test", run.name)
    else:
        output_root = os.path.join(output_dir, "train")
    if os.path.exists(output_root):
        print("✅ Folder with relevent files exists")
        pass
    else:
        os.makedirs(output_root)
        if test:
            name_batch = "Merged_test_Batch_prompt_data"
        else:
            name_batch = "Merged_Batch_prompt_data"
        for artifact_ref in run.logged_artifacts():
            if name_batch in artifact_ref.name:
                print(f"Scanning: {artifact_ref.name}")
                number_tries = 0
                while number_tries < 5:
                    try:
                        for file in artifact_ref.files():
                            if file.name.endswith(".json"):
                                output_path = os.path.join(output_root, f"{artifact_ref.name}_{file.name}")
                                print(f"Downloading {file.name} -> {output_path}")
                                file.download(root=output_root, replace=True)
                        break

                    except Exception as e:
                        print(f"❌ Failed to process {artifact_ref.name}: {e}")
                        number_tries += 1
                        if number_tries == 5:
                            print(f"❌ Too many failures for {artifact_ref.name}. Skipping.")
                            break

    print("✅ Done downloading all relevant artifacts.")
    return output_root

def create_csv_from_json_files(output_root, return_df=False):
    all_dfs = []
    for filename in os.listdir(output_root):
        if filename.endswith(".json"):
            file_path = os.path.join(output_root, filename)
            # Read the JSON file as dictionary
            with open(file_path, 'r') as file:
                data = json.load(file)
            # Convert the dictionary to a DataFrame
            all_dfs.append(pd.DataFrame(data['data'], columns=data['columns']))

    # Concatenate all DataFrames into a single DataFrame
    merged_df = pd.concat(all_dfs, ignore_index=True)
    # Save the merged DataFrame to a CSV file
    merged_df.to_csv(os.path.join(output_root, "merged_data.csv"), index=False)
    if return_df:
        return merged_df

def main(config_file):
    # Load the config file
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)

    train_id_experiment = config['train_id_experiment']
    test_ids_experiment = config['test_id_experiment']
    if isinstance(test_ids_experiment, str):
        test_ids_experiment = [test_ids_experiment]

    logging.info(f"Downloading files from train experiment: {train_id_experiment}")
    train_output_root = download_files(train_id_experiment, train_id_experiment, config['output_dir'], test=False)
    create_csv_from_json_files(train_output_root, return_df=False)

    list_dfs = []
    for test_id_experiment in test_ids_experiment:
        logging.info(f"Downloading files from test experiment: {test_id_experiment}")
        test_output_root_k = download_files(test_id_experiment, test_id_experiment, config['output_dir'], test=True)
        list_dfs.append(create_csv_from_json_files(test_output_root_k, return_df=True))
    merged_test_df = pd.concat(list_dfs, ignore_index=True)
    merged_test_df.to_csv(os.path.join(config['output_dir'], "test", "merged_test_data.csv"), index=False)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download experiment files from W&B.")
    parser.add_argument("--config_file", type=str, required=True, help="Path to the config file.")
    args = parser.parse_args()

    main(args.config_file)

