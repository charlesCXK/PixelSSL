import os
import sys
import pprint
import argparse
import azureml.core
from azureml.core import Workspace
from azureml.core import Datastore
from azureml.core import Experiment
from azureml.core.compute import ComputeTarget
from azureml.core.runconfig import ContainerRegistry
from azureml.train.dnn import PyTorch

from azureml.train.estimator import Estimator
from azureml.core.runconfig import MpiConfiguration

'''
Choose training cluster by use different configs
'''
import config_ussc as config

if __name__ == '__main__':

    '''
    Submit a job by execute:
      python run_docker_inst.py --cfg xxx
    The xxx will be passed into the 'entry_script', and will also be displayed as the 'tag'
    '''
    parser = argparse.ArgumentParser(description="AML Generic Launcher")
    parser.add_argument("--cfg", default="")
    args, _ = parser.parse_known_args()

    # docker image registry, no need to change if you want to use Philly docker
    container_registry_address = "phillyregistry.azurecr.io/" # example : "phillyregistry.azurecr.io"
    container_registry_username = ""
    container_registry_password = ""
    # custom_docker_image ="philly/jobs/custom/pytorch:v1.1-py36-hrnet" # example: "philly/jobs/custom/pytorch:your tag"

    '''
    AML can use Docker images in the DockerHub, specify your Docker image here
    '''
    custom_docker_image = "charlescxk/ssc:1.0" # example: "philly/jobs/custom/pytorch:your tag" "pytorch/pytorch:1.5-cuda10.1-cudnn7-devel"


    '''
    entry_script: Specify the script you want to execute, here I set to be ./docker/inst_efficienthrnet.py as default script
    '''
    # Note: source_directory and entry_script are in local, source_directory/entry_script
    source_directory = "./docker"
    # print(sys.argv[1])
    # entry_script = sys.argv[1]
    entry_script = 'cxk.py'
    # entry_script = "./entry-script.py"

    subscription_id = config.subscription_id
    resource_group = config.resource_group
    workspace_name = config.workspace_name
    ws = Workspace(subscription_id = subscription_id, resource_group = resource_group, workspace_name = workspace_name)

    cluster_name= config.cluster_name
    ct = ComputeTarget(workspace=ws, name=cluster_name)
    datastore_name =config.datastore_name
    ds = Datastore(workspace=ws, name=datastore_name)

    script_params = {
      "--workdir": ds.path('./').as_mount(), # REQUIRED !!!
      "--cfg" : args.cfg
    }

    def make_container_registry(address, username, password):
        cr = ContainerRegistry()
        cr.address = address
        cr.username = username
        cr.password = password
        return cr

    my_registry = make_container_registry(
            address=container_registry_address,
            username=container_registry_username,
            password=container_registry_password )

    estimator = PyTorch(source_directory='./docker',
                          script_params=script_params,
                          compute_target=ct,
                          use_gpu=True,
                          shm_size='256G',
                          # image_registry_details= my_registry, 
                          entry_script=entry_script,        
                          custom_docker_image=custom_docker_image,
                          user_managed=True
                        )

    experiment = Experiment(ws, name=config.experiment_name)

    run = experiment.submit(estimator, tags={'tag': args.cfg})

    pprint.pprint(run)

    # uncomment next line to see the stdout in your main.py on local machine.
    #run.wait_for_completion(show_output=True) 

    # (END)
