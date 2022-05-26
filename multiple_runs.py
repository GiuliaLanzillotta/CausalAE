""" Script responsible for the generation of multiple experiments config files based on a given base model
and obtained by varying one of the attributes"""
import copy
from typing import List

import yaml
from pathlib import Path


def generate_multiple_configs(model_names:List[str], model_versions:List[str], attribute_family:str,
                              attribute_name:str, attribute_values:list, command:str=None):
    """Generates multiple config files - one for each value in the 'attribute values list- following the specifics of the
    selected model (name and version are given) and only changing the selected attribute
    We will also record the bash commands to launch the different jobs on the cluster based on the command
    for the base model 'command' if provided
    @model_names: list of models to apply the generation to
    @model_versions: list of versions, for each model
    """

    # initialise list of commands
    write_commands = not (command is None)
    if write_commands: commands = []

    for model_name in model_names:
        #first get the original config file
        base_path = Path('configs/models')/model_name
        if write_commands:
            base_command = copy.deepcopy(command)
            base_command = base_command.replace('_name',model_name)

        for model_version in model_versions:
            print(f"Generating material for multiple runs on {model_name} {model_version} for {attribute_name}")

            base_config_path = str(base_path) + "/" + model_version +'.yaml'
            with open(base_config_path, 'r') as file:
                fig = yaml.safe_load(file)

            # now create and save a new config file for each different value
            for v in attribute_values:
                fig[attribute_family][attribute_name] = v
                new_version_name = model_version+attribute_name+str(v)
                new_path = str(base_path) + "/" + new_version_name +'.yaml'
                with open(new_path, 'w') as out:
                    yaml.dump(fig, out, default_flow_style=False)
                if write_commands:
                    new_command = copy.deepcopy(base_command)
                    new_command = new_command.replace('_version',new_version_name)
                    #replacing the title of the job as well
                    job_signature =  model_version[-1]+attribute_name[0]+str(v)
                    new_command = new_command.replace('title',job_signature)
                    commands.append(new_command)
            commands.append("\n")
        commands.append("\n")

    if write_commands:
        #now save all the commands in the related file
        full_Text = str.join("\n", commands)
        text_file_name = "jobs_"+attribute_name+".txt"
        #notice that the file will be written in the cwd
        with open(text_file_name, "a") as text_file:
            text_file.write(full_Text)
            text_file.write("\n") # final space at the end



if __name__ == '__main__':
    _model_names = ["CausalVAE"]
    _model_versions = ["fc_pd","fc_pd_1D"]
    _attribute_family = "model_params"
    _attribute_name = "random_seed"
    _attribute_values = [13,17,37,121]
    _command = 'bsub -oo "_name_title_Pendulum.txt" -R "rusage[mem=30000, ngpus_excl_p=1]" ' \
              'python main.py --name _name --data Pendulum --version _version'
    generate_multiple_configs(_model_names, _model_versions, _attribute_family,
                              _attribute_name, _attribute_values, _command)