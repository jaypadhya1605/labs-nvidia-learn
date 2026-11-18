using './main.bicep'

// Values come from env/lab.env (exported by infra/deploy-landing-zone.sh)
param prefix = readEnvironmentVariable('PREFIX', 'khalab')
param location = readEnvironmentVariable('LOCATION', 'eastus')
param userPrincipalId = readEnvironmentVariable('USER_PRINCIPAL_ID', '')
param kubernetesVersion = readEnvironmentVariable('K8S_VERSION', '')
