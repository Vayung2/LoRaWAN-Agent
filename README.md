To run the traditional approach:
1. python -m src.build_metadata  
2. python -m src.traditional.calibrate  --data_dir dataset/lorawan_metadata  --meta models/metadata.json   --out models/traditional_params.json  
(once ^^)
3. python -m src.run_inference --method traditional --data_dir dataset/lorawan_metadata --out_csv reports/loc_traditional.csv  

To run the ML approach:
1. python -m src.build_metadata  
2. python -m src.regression_model.train --data_dir dataset/lorawan_metadata --out_model models/regression_model.joblib
3. python -m src.run_inference --method regression --data_dir dataset/lorawan_metadata --out_csv reports/loc_regression.csv

To run both approaches under the:
1. foil attack (RSSI shifts negative. Right now, im doing this for all RSSI, not just individual sensor):
python -m src.run_attack_sweep --shifts "0,-3,-6,-10,-15,-20" --methods "traditional,regression" --out_dir reports/attacks/foil_shift


