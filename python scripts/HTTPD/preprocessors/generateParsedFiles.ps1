Write-Host "[*] Generating benign ..."
python .\log2csv.py "..\datasets\originals\benign.csv"
python .\appendFlag2CSV.py "..\datasets\parsed\parsed_benign.csv" "..\datasets\parsed\parsed_benign.csv" flag normal

Write-Host "[*] Generating ffuf-extensions ..."
python .\log2csv.py "..\datasets\originals\ffuf-extensions.csv"
python .\appendFlag2CSV.py "..\datasets\parsed\parsed_ffuf-extensions.csv" "..\datasets\parsed\parsed_ffuf-extensions.csv" flag anomaly

Write-Host "[*] Generating ffuf-small ..."
python .\log2csv.py "..\datasets\originals\ffuf-small.csv"
python .\appendFlag2CSV.py "..\datasets\parsed\parsed_ffuf-small.csv" "..\datasets\parsed\parsed_ffuf-small.csv" flag anomaly

Write-Host "[*] Generating httpd_logs_with_behavior ..."
python .\httpd_logs_with_behavior_parser.py

Write-Host "[*] Generating search_benign ..."
python .\appendFlag2CSV.py "..\datasets\originals\search_benign.csv" "..\datasets\parsed\parsed_search_benign.csv" flag normal

Write-Host "[*] Generating xss_exploit_logs ..."
python .\appendFlag2CSV.py "..\datasets\originals\xss_exploit_logs.csv" "..\datasets\parsed\parsed_xss_exploit_logs.csv" flag anomaly

Write-Host "[*] FUSING PARSED CSVs together..."
python .\fuseAllCSVs.py "..\datasets\parsed" "..\datasets\parsed\HTTPD_FULL_TRAINING_DATA.CSV"

Write-Host "[*] Generating LABELLESS CSVs ..."
python .\labelless_csv_generator.py "..\datasets\parsed\" "..\datasets\parsed\labelless" 

Write-Host "[*] FUSING PARSED CSVs together..."
python .\fuseAllCSVs.py "..\datasets\parsed\labelless" "..\datasets\parsed\labelless\HTTPD_FULL_TRAINING_DATA_LABELLESS.CSV"


Write-Host "[*] DONE!"
