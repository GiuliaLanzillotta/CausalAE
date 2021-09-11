""" Script responsible for the generation of multiple experiments config files based on a given base model
and obtained by varying one of the attributes"""

import yaml
from pathlib import Path


def generate_multiple_configs(model_name:str, model_version:str, attribute_family:str,
                              attribute_name:str, attribute_values:list, command:str=None):
    """Generates multiple config files - one for each value in the 'attribute values list- following the specifics of the
    selected model (name and version are given) and only changing the selected attribute
    We will also record the bash commands to launch the different jobs on the cluster based on the command
    for the base model 'command' if provided

    #TODO: expand with possibility to add multiple models (same versions to it)"""

    print(f"Generating material for multiple runs on {model_name} {model_version} for {attribute_name}")

    # initialise list of commands
    write_commands = not (command is None)
    if write_commands: commands = []


    #first get the original config file
    base_path = Path('configs/models')/model_name
    base_config_path = str(base_path) + "/" + model_version +'.yaml'
    with open(base_config_path, 'r') as file:
        fig = yaml.safe_load(file)

    # now create and save a new config file for each different value
    for v in attribute_values:
        fig[attribute_family][attribute_name] = v
        new_version_name = attribute_name+str(v)
        new_path = str(base_path) + "/" + new_version_name +'.yaml'
        with open(new_path, 'w') as out:
            yaml.dump(fig, out, default_flow_style=False)
        if write_commands:
            new_command = command.replace(model_version, new_version_name)
            #replacing the title of the job as well
            job_signature =  model_version[-1]+attribute_name[0]+str(v)
            new_command = new_command.replace('title',job_signature)
            commands.append(new_command)

    if write_commands:
        #now save all the commands in the related file
        full_Text = str.join("\n", commands)
        text_file_name = "jobs_"+attribute_name+".txt"
        #notice that the file will be written in the cwd
        with open(text_file_name, "a") as text_file:
            text_file.write(full_Text)
            text_file.write("\n") # final space at the end



if __name__ == '__main__':
    _model_name = "XCAE"
    _model_version = "standardS"
    _attribute_family = "model_params"
    _attribute_name = "xunit_dim"
    _attribute_values = range(3,8)
    _command = 'bsub -oo "XCAE_title_MNIST.txt" -W 4:00 -R "rusage[mem=30000, ngpus_excl_p=1]" ' \
              'python main.py --name XCAE --data MNIST --version standardS'
    generate_multiple_configs(_model_name, _model_version, _attribute_family,
                              _attribute_name, _attribute_values, _command)