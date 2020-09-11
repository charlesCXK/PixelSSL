#
# AML configurations, which can be found in AML page on Azure Portal
#
subscription_id = "db9fc1d1-b44e-45a8-902d-8c766c255568"
resource_group  = "koreav100"
workspace_name  = "koreav100ws"

#
# Azure storage blob configurations, which can be found in Storage page on Azure Portal
#
blob_container_name = "storage1"
blob_account_name   = "depu"
blob_account_key    = "1H1Hqzc3QZpBacsXqpwi/SOFIAjWAfVt5806Q0Ns8Ymz+iJ6yOj/k2KJS2E23S84rZRjLyCSuqC3+NlkQpUiJw=="

#
# Azure storage file configurations, which can be found in Storage page on Azure Portal
#
file_container_name = "rainbowsecret"
file_account_name   = "openseg"
file_account_key    = "i0Gj/Itb1NUJm0KyMPouUF/Opr2c4QBUqMynOU7k0WKPvzFJLEVGPjklFKPJOBr5ZMVdXUY2Qq/2PIruY9I9Bw=="

#
# Job detail configuration
#
datastore_name  = "depudatastore_v2"     # You can use any name here.
experiment_name = "aml-depu-test"    # You can use any name here.
cluster_name    = "korea24cl"
