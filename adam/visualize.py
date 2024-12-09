import os
import sys
import h5py
import matplotlib.pyplot as plt
import pandas as pd


def main(folder=None):
    # Step 1: Get folder location from command-line argument or use current directory
    folder = folder if folder else os.getcwd()

    # Step 2: Check each subfolder for `training/result.h5`
    subfolders = []
    for subdir, _, _ in os.walk(folder):
        if os.path.isfile(os.path.join(subdir, 'training/result.h5')):
            subfolders.append(subdir)

    # Step 3: Compose dataframe of mean local energy for all folders
    energy_data = {}
    for subfolder in subfolders:
        h5_file = os.path.join(subfolder, 'training/result.h5')
        try:
            with h5py.File(h5_file, 'r', swmr=True) as f:
                energy = f['local_energy']['mean'][:]
                energy_data[subfolder] = energy[:, 0, 0]
        except Exception as e:
            print(f"Error reading {h5_file}: {e}")

    # Create DataFrame
    energy_df = pd.DataFrame.from_dict(energy_data, orient='index').transpose()

    # Step 4: Plot all folder data
    plt.figure(figsize=(10, 6))
    for folder_name, energy_series in energy_df.items():
        #plt.plot(energy_series, label=f"Raw {folder_name}")
        plt.plot(pd.DataFrame(energy_series).ewm(halflife=5).mean(), label=f"Smoothed {folder_name}")

    # Set the y scale to log
    # plt.yscale('log')

    plt.xlabel('Training Iteration')
    plt.ylabel('Energy')
    plt.legend()
    plt.title('Energy Trends')
    plt.show()


if __name__ == "__main__":
    # Command-line argument to specify the folder
    specified_folder = sys.argv[1] if len(sys.argv) > 1 else None
    main(folder=specified_folder)
