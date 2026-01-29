import subprocess
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
import numpy as np

# Configuration
model_name = "MicroreactorPK.Blocks.DrumProfileFromFile"
mat_file = os.path.abspath("../../../drumv4out.mat")
output_csv = "results.mat"

# 1. Compile the block
print("--- Compiling Modelica Block ---")
subprocess.run(["omc", "export.mos"], check=True)

# 2. Define the path to the Linux binary
# On Linux, buildModel creates a file with the exact class name
binary_path = os.path.abspath(f".build/{model_name}")

# Ensure the binary is executable (OMC usually does this, but just in case)
os.chmod(binary_path, 0o755)

# 3. Run the binary directly
# -override: Replaces the fileName parameter
# -r: Sets the output filename
# -format: Forces CSV
print(f"--- Running Simulation with {mat_file} ---")
try:
    subprocess.run([
        f"{binary_path}", 
        "-override", f"fileName={mat_file}",
        "-r", output_csv
    ], check=True, cwd=os.path.join(os.getcwd(), ".build"))
except subprocess.CalledProcessError as e:
    print(f"Simulation failed. Error code: {e.returncode}")
    # Check the output log if it exists
    log_file = f"{model_name}_log.txt"
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            print(f.read())

# # 4. Plotting
def decode_names(name_matrix):
    variable_names_proper = []
    #FIRST PAD WITH ZEROS
    pad_template = max(name_matrix, key=len)
    for name in name_matrix:
        if len(name) != len(pad_template):
            name_matrix[np.where(name_matrix == name)] = str(name)+("0"*(len(pad_template)-len(name)))
    #LINE THEM UP (use transpose)
    for col in range(0,len(pad_template)):
        string_form = ""
        for row in range(0,len(name_matrix)):
            string_form += name_matrix[row][col]
        variable_names_proper.append(string_form.replace('0','').replace('\x00', ''))
        
    return variable_names_proper

if os.path.exists(os.path.join(".build", output_csv)):
    data_res = loadmat(os.path.join(".build", output_csv))
    # print(decode_names(data_res['name'])) # ['time', 'angleDeg', 'angleColumn']
    plt.plot(data_res['data_2'][0], data_res['data_2'][1], label="angledeg vs time", color="blue")
    # plt.plot(data_res['data_2'][0], data_res['data_2'][2], label="anglecolumn vs time", color="red")
    plt.title("angle stuff")
    plt.xlabel("time")
    plt.ylabel("angles")
    #plt.grid(True, linestyle="--",alpha=0.7)
    plt.legend()
    plt.show()