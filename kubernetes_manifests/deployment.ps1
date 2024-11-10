Write-Host "[*] Cleaning existing..."
kubectl delete all --all -n default
Write-Host "[*] Setting route..."
Set-Location C:\Users\Fran-Trabajo\Documents\Master\UNIR\SevenSector\my-digital-twin\kubernetes_manifests
Write-Host "[*] Deploying Persisten Volume Claims..."
kubectl apply -f .\ftp-server\ftp-pvc.yaml
kubectl apply -f .\sftp-server\sftp-pvc.yaml
Write-Host "[*] Deploying FTP Server ..."
kubectl apply -f .\ftp-server\ftp-server.yaml
Write-Host "[*] Deploying SFTP Server ..."
kubectl apply -f .\sftp-server\sftp-server.yaml
Write-Host "[*] Deploying Vulnerable MicroService ..."
kubectl apply -f .\vulnerable-microservice\ms-vulnerable-gcloud.yaml
Write-Host "[*] Deploying Load Balancer ..."
kubectl apply -f .\load-balancer.yaml