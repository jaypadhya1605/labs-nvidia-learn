output "public_ip" { value = azurerm_public_ip.pip.ip_address }
output "ssh" { value = "ssh plvadmin@${azurerm_public_ip.pip.ip_address}" }
