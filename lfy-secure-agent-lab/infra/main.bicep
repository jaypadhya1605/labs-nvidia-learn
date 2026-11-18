// Lanternfly Security - Secure Agentic AI Lab (Lab 03). Same landing zone as Lab 02, plus ACR and Cilium network policy.
// Creates: AKS (system pool only), Key Vault (RBAC), Azure Monitor workspace, Azure Managed Grafana, optional ACR.
// GPU node pools are added by infra/deploy-landing-zone.sh with `--gpu-driver none` so the NVIDIA GPU Operator owns the driver.

@description('Short lowercase prefix for resource names (3-10 chars).')
@minLength(3)
@maxLength(10)
param prefix string = 'lfylab'

param location string = resourceGroup().location

@description('Object ID of the person running the lab (az ad signed-in-user show --query id -o tsv).')
param userPrincipalId string

@description('Kubernetes version. Empty string = AKS default.')
param kubernetesVersion string = ''

param systemVmSize string = 'Standard_D8ds_v5'
param systemNodeCount int = 2
param deployGrafana bool = true
param deployAcr bool = true   // Lab 03 builds and signs its own images

param tags object = {
  project: 'lfy-secure-agent-lab'
  purpose: 'nvidia-sa-training'
  data: 'synthetic-only'
}

var suffix = uniqueString(resourceGroup().id)
var kvName = take('${prefix}kv${suffix}', 24)
var acrName = take('${prefix}acr${suffix}', 50)
var aksName = '${prefix}-aks'
var amwName = '${prefix}-amw'
var grafanaName = take('${prefix}-grafana', 23)

// Built-in role definition IDs
var roleKeyVaultSecretsOfficer = 'b86a8fe4-44ce-4948-aee5-eccb2c155cd7'
var roleGrafanaAdmin = '22926164-76b3-42b3-bc55-97df8dab3e41'
var roleMonitoringDataReader = 'b0d8363b-8ddd-447d-831f-62ca05bff136'
var roleAcrPull = '7f951dda-4ed3-4680-a7ca-43fe172d538d'

resource kv 'Microsoft.KeyVault/vaults@2023-07-01' = {
  name: kvName
  location: location
  tags: tags
  properties: {
    tenantId: subscription().tenantId
    sku: {
      family: 'A'
      name: 'standard'
    }
    enableRbacAuthorization: true
    enableSoftDelete: true
    softDeleteRetentionInDays: 7
    publicNetworkAccess: 'Enabled'
  }
}

resource kvUserSecretsOfficer 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(kv.id, userPrincipalId, roleKeyVaultSecretsOfficer)
  scope: kv
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roleKeyVaultSecretsOfficer)
    principalId: userPrincipalId
    principalType: 'User'
  }
}

// NEW in Lab 03 (M11): sign container images with a Key Vault certificate via Notation
var roleKeyVaultCertificatesOfficer = 'a4417e6f-fecd-4de8-b567-7b0420556985'
var roleKeyVaultCryptoUser = '12338af0-0e69-4776-bea7-57ae8d297424'
resource kvUserCertOfficer 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(kv.id, userPrincipalId, roleKeyVaultCertificatesOfficer)
  scope: kv
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roleKeyVaultCertificatesOfficer)
    principalId: userPrincipalId
    principalType: 'User'
  }
}
resource kvUserCryptoUser 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(kv.id, userPrincipalId, roleKeyVaultCryptoUser)
  scope: kv
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roleKeyVaultCryptoUser)
    principalId: userPrincipalId
    principalType: 'User'
  }
}

resource amw 'Microsoft.Monitor/accounts@2023-04-03' = {
  name: amwName
  location: location
  tags: tags
}

resource grafana 'Microsoft.Dashboard/grafana@2023-09-01' = if (deployGrafana) {
  name: grafanaName
  location: location
  tags: tags
  sku: {
    name: 'Standard'
  }
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    publicNetworkAccess: 'Enabled'
    zoneRedundancy: 'Disabled'
    apiKey: 'Disabled'
    grafanaIntegrations: {
      azureMonitorWorkspaceIntegrations: [
        {
          azureMonitorWorkspaceResourceId: amw.id
        }
      ]
    }
  }
}

resource grafanaReadsAmw 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (deployGrafana) {
  name: guid(amw.id, grafanaName, roleMonitoringDataReader)
  scope: amw
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roleMonitoringDataReader)
    principalId: grafana!.identity.principalId
    principalType: 'ServicePrincipal'
  }
}

resource grafanaUserAdmin 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (deployGrafana) {
  name: guid(grafanaName, userPrincipalId, roleGrafanaAdmin)
  scope: grafana
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roleGrafanaAdmin)
    principalId: userPrincipalId
    principalType: 'User'
  }
}

resource aks 'Microsoft.ContainerService/managedClusters@2024-09-01' = {
  name: aksName
  location: location
  tags: tags
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    dnsPrefix: '${prefix}-dns'
    kubernetesVersion: empty(kubernetesVersion) ? null : kubernetesVersion
    enableRBAC: true
    oidcIssuerProfile: {
      enabled: true
    }
    securityProfile: {
      workloadIdentity: {
        enabled: true
      }
    }
    networkProfile: {
      networkPlugin: 'azure'
      networkPluginMode: 'overlay'
      networkDataplane: 'cilium'   // NEW in Lab 03: enforces Kubernetes NetworkPolicy (agent egress lock-down)
      networkPolicy: 'cilium'
      loadBalancerSku: 'standard'
      podCidr: '10.244.0.0/16'
      serviceCidr: '10.0.0.0/16'
      dnsServiceIP: '10.0.0.10'
    }
    agentPoolProfiles: [
      {
        name: 'system'
        mode: 'System'
        count: systemNodeCount
        vmSize: systemVmSize
        osType: 'Linux'
        osSKU: 'Ubuntu'
        type: 'VirtualMachineScaleSets'
        maxPods: 110
      }
    ]
  }
}

resource acr 'Microsoft.ContainerRegistry/registries@2023-07-01' = if (deployAcr) {
  name: acrName
  location: location
  tags: tags
  sku: {
    name: 'Basic'
  }
  properties: {
    adminUserEnabled: false
  }
}

resource kubeletPullsAcr 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (deployAcr) {
  name: guid(acrName, aksName, roleAcrPull)
  scope: acr
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roleAcrPull)
    principalId: aks.properties.identityProfile.kubeletidentity.objectId
    principalType: 'ServicePrincipal'
  }
}

output aksName string = aks.name
output kvName string = kv.name
output amwId string = amw.id
output grafanaName string = deployGrafana ? grafanaName : ''
output grafanaEndpoint string = deployGrafana ? grafana!.properties.endpoint : ''
output kubeletClientId string = aks.properties.identityProfile.kubeletidentity.clientId
output kubeletObjectId string = aks.properties.identityProfile.kubeletidentity.objectId
output acrName string = deployAcr ? acrName : ''
