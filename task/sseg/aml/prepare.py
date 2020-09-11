import os
import sys
import azureml.core
from azureml.core import Workspace
from azureml.core import Datastore
import config_ussc as config

subscription_id = config.subscription_id
resource_group = config.resource_group
workspace_name = config.workspace_name

datastore_name = config.datastore_name

# blob_container_name = config.file_container_name
# blob_account_name = config.file_account_name
# blob_account_key = config.file_account_key
 
blob_container_name = config.blob_container_name
blob_account_name = config.blob_account_name
blob_account_key = config.blob_account_key
 
#
# Prepare the workspace.
#
ws = None
try:
    print("Connecting to workspace '%s'..." % workspace_name)
    ws = Workspace(subscription_id = subscription_id, resource_group = resource_group, workspace_name = workspace_name)
except:
    print("Workspace not accessible.")
print(ws.get_details())

ws.write_config()

#
# Register an existing datastore to the workspace.
#
if datastore_name not in ws.datastores:
    # Datastore.register_azure_file_share(
    #     workspace=ws, 
    #     datastore_name=datastore_name,
    #     file_share_name=blob_container_name,
    #     account_name=blob_account_name,
    #     account_key=blob_account_key
    Datastore.register_azure_blob_container(
        workspace=ws, 
        datastore_name=datastore_name,
        container_name=blob_container_name,
        account_name=blob_account_name,
        account_key=blob_account_key
    )
    print("Datastore '%s' registered." % datastore_name)
else:
    print("Datastore '%s' has already been regsitered." % datastore_name)

# (END)
