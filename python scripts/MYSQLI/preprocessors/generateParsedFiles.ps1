Write-Host "[**] Generating labelless ..."
Write-Host "[*] Generating benign ..."
python .\parseLog.py ..\datasets\originals\normal_queries_with_prefix.csv
Write-Host "[*] Generating SQLi attacks ..."
python .\parseLog.py ..\datasets\originals\final_injection_queries_with_prefix.csv
Write-Host "[**] Generating parsed ..."
Write-Host "[*] Generating benign ..."
python .\parseLog.py ..\datasets\originals\normal_queries_with_prefix.csv label normal
Write-Host "[*] Generating SQLi attacks ..."
python .\parseLog.py ..\datasets\originals\final_injection_queries_with_prefix.csv label sqli_attack
Write-Host "[**] Generating final testing dataset ..."
python .\testingDataGenerator.py
Write-Host "[**] DONE!"
