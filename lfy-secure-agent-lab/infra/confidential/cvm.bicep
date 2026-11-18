// M12 (optional): confidential GPU VM (AMD SEV-SNP CPU TEE + H100 in confidential-computing mode) and a Key Vault
// Premium key that Azure releases ONLY to a VM that proves - via Microsoft Azure Attestation - it is an Azure-compliant CVM.
// Needs NCCadsH100v5 quota. eastus2 price (2026-10-04): $6.98/h PAYG, ~$1.29/h Spot.
param location string = 'eastus2'
@minLength(3)
@maxLength(10)
param prefix string = 'lfycvm'
param adminUsername string = 'azureuser'
@secure()
param sshPublicKey string
param allowedSshCidr string
param userPrincipalId string
param vmSize string = 'Standard_NCC40ads_H100_v5'
param spot bool = true
param maaEndpoint string = 'https://sharedeus2.eus2.attest.azure.net'

var kvName = take('${prefix}kv${uniqueString(resourceGroup().id)}', 24)
var roleCryptoServiceReleaseUser = '08bbd89e-9f13-488c-ac41-acfcb10c90ab'
var roleKeyVaultCryptoUser = '12338af0-0e69-4776-bea7-57ae8d297424'
var releasePolicy = {
  version: '1.0.0'
  anyOf: [
    {
      authority: maaEndpoint
      allOf: [
        { claim: 'x-ms-isolation-tee.x-ms-attestation-type', equals: 'sevsnpvm' }
        { claim: 'x-ms-isolation-tee.x-ms-compliance-status', equals: 'azure-compliant-cvm' }
      ]
    }
  ]
}

resource nsg 'Microsoft.Network/networkSecurityGroups@2023-11-01' = {
  name: '${prefix}-nsg'
  location: location
  properties: {
    securityRules: [
      {
        name: 'ssh-from-you-only'
        properties: {
          priority: 100
          direction: 'Inbound'
          access: 'Allow'
          protocol: 'Tcp'
          sourceAddressPrefix: allowedSshCidr
          sourcePortRange: '*'
          destinationAddressPrefix: '*'
          destinationPortRange: '22'
        }
      }
    ]
  }
}

resource vnet 'Microsoft.Network/virtualNetworks@2023-11-01' = {
  name: '${prefix}-vnet'
  location: location
  properties: {
    addressSpace: { addressPrefixes: ['10.20.0.0/16'] }
    subnets: [
      {
        name: 'cvm'
        properties: { addressPrefix: '10.20.1.0/24', networkSecurityGroup: { id: nsg.id } }
      }
    ]
  }
}

resource pip 'Microsoft.Network/publicIPAddresses@2023-11-01' = {
  name: '${prefix}-pip'
  location: location
  sku: { name: 'Standard' }
  properties: { publicIPAllocationMethod: 'Static' }
}

resource nic 'Microsoft.Network/networkInterfaces@2023-11-01' = {
  name: '${prefix}-nic'
  location: location
  properties: {
    ipConfigurations: [
      {
        name: 'ipconfig1'
        properties: {
          subnet: { id: vnet.properties.subnets[0].id }
          publicIPAddress: { id: pip.id }
          privateIPAllocationMethod: 'Dynamic'
        }
      }
    ]
  }
}

resource vm 'Microsoft.Compute/virtualMachines@2024-07-01' = {
  name: '${prefix}-vm'
  location: location
  identity: { type: 'SystemAssigned' }
  properties: {
    priority: spot ? 'Spot' : 'Regular'
    evictionPolicy: spot ? 'Deallocate' : null
    billingProfile: spot ? { maxPrice: -1 } : null
    hardwareProfile: { vmSize: vmSize }
    securityProfile: {
      securityType: 'ConfidentialVM'
      uefiSettings: { secureBootEnabled: true, vTpmEnabled: true }
    }
    storageProfile: {
      imageReference: { publisher: 'Canonical', offer: 'ubuntu-24_04-lts', sku: 'cvm', version: 'latest' }
      osDisk: {
        createOption: 'FromImage'
        diskSizeGB: 256
        managedDisk: {
          storageAccountType: 'Premium_LRS'
          securityProfile: { securityEncryptionType: 'VMGuestStateOnly' }
        }
      }
    }
    osProfile: {
      computerName: '${prefix}-vm'
      adminUsername: adminUsername
      linuxConfiguration: {
        disablePasswordAuthentication: true
        ssh: { publicKeys: [ { path: '/home/${adminUsername}/.ssh/authorized_keys', keyData: sshPublicKey } ] }
      }
    }
    networkProfile: { networkInterfaces: [ { id: nic.id } ] }
  }
}

resource kv 'Microsoft.KeyVault/vaults@2023-07-01' = {
  name: kvName
  location: location
  properties: {
    tenantId: subscription().tenantId
    sku: { family: 'A', name: 'premium' }
    enableRbacAuthorization: true
    enableSoftDelete: true
    softDeleteRetentionInDays: 7
    enablePurgeProtection: true
  }
}

resource modelKey 'Microsoft.KeyVault/vaults/keys@2023-07-01' = {
  parent: kv
  name: 'lfy-model-key'
  properties: {
    kty: 'RSA-HSM'
    keySize: 3072
    keyOps: ['wrapKey', 'unwrapKey', 'encrypt', 'decrypt']
    attributes: { enabled: true, exportable: true }
    release_policy: {
      contentType: 'application/json; charset=utf-8'
      data: base64(string(releasePolicy))
    }
  }
}

resource vmCanRelease 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(kv.id, vm.id, roleCryptoServiceReleaseUser)
  scope: kv
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roleCryptoServiceReleaseUser)
    principalId: vm.identity.principalId
    principalType: 'ServicePrincipal'
  }
}

resource userCanEncrypt 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(kv.id, userPrincipalId, roleKeyVaultCryptoUser)
  scope: kv
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roleKeyVaultCryptoUser)
    principalId: userPrincipalId
    principalType: 'User'
  }
}

output publicIp string = pip.properties.ipAddress
output kvName string = kv.name
output keyKid string = modelKey.properties.keyUriWithVersion
output maaEndpoint string = maaEndpoint
