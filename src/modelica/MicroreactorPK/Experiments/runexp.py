import os
import numpy as np
import scipy.io
import OMSimulator as oms
import matplotlib.pyplot as plt

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


fmu = oms.FMU('./RunOneProfile.fmu')
print("name: ", fmu.modelName)
print("guid: ", fmu.guid)
print("fmi version: ", fmu.fmiVersion)

print("states: ")
for var in sorted(fmu.states, key=lambda x:x.name):
    print({
        'name': var.name,
        'signal_type': var.signal_type.name,
        'valueReference': var.valueReference,
        'variability': var.variability,
        'causality': var.causality.name
    })

fmu.instantiate()
fmu.setValue("profileFile", os.path.abspath("../../../drumv4out.mat").encode('utf-8'))
fmu.setResultFile('drum_profile_output_localsim.mat')
fmu.initialize()
fmu.simulate()
fmu.terminate()
fmu.delete()

print(f"{bcolors.OKGREEN}Simulation complete!{bcolors.ENDC}")

data=scipy.io.loadmat('drum_profile_output_localsim.mat')
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
print(f"{bcolors.OKCYAN}{decode_names(data['name'])}{bcolors.ENDC}")

plt.plot(data['data_2'][0], data['data_2'][2], label="nitrogen temperature", color='blue')
plt.plot(data['data_2'][0], data['data_2'][1], label="fuel temperature", color="red", linestyle="--")
plt.title("just the first two datas")
plt.xlabel("time s")
plt.ylabel("temps")
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend()
plt.show()