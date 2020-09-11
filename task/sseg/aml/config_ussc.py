#
# AML configurations, which can be found in AML page on Azure Portal
#
subscription_id = "d4c59fc3-1b01-4872-8981-3ee8cbd79d68"
resource_group  = "usscv100"
workspace_name  = "usscv100ws"


'''
Change the blob configuration to yours here
'''
#
# Azure storage blob configurations, which can be found in Storage page on Azure Portal
#
blob_container_name = "pretrain"
blob_account_name   = "cxk"
blob_account_key    = "YbLOkA3pNqJUWs5W/5R6D0B3dLkGFYNcRL+1KxbBxw/gPf5ZOrcMYDuxio39et8+0tHgWWsGSw5jdUJFbuwyMQ=="

'''
datastore_name:     Give this Cluster-Blob bidding a name, this name does not matter
experiment_name:    The name of your experiment, I use the same experiment name for all my experiments
                        So that all my jobs will be clustered together and easy to find.
cluster_name:       Usually not need to change
'''
#
# Job detail configuration
#
datastore_name  = "cxk_datastore"     # You can use any name here.
experiment_name = "ssslearning"    # You can use any name here.
cluster_name    = "usscv100cl"
