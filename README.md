# ValuAItion
Predicting valuation prices of danish properties using machine learning.

Project started as Hackathon by Resights posted on LinkedIn:
- [Snowflake Dataset](https://app.snowflake.com/marketplace/listing/GZSYZP5GJV/resights-aps-resights-avm)
- [Initial Post](https://www.linkedin.com/posts/mikkelduif_hackathon-kan-du-sl%C3%A5-statens-ejendomsvurderingsmodel-activity-7249316341243891726--hK6)
- [Final Post](https://www.linkedin.com/posts/mikkelduif_%F0%9D%90%87%F0%9D%90%9A%F0%9D%90%9C%F0%9D%90%A4%F0%9D%90%9A%F0%9D%90%AD%F0%9D%90%A1%F0%9D%90%A8%F0%9D%90%A7-%F0%9D%90%96%F0%9D%90%A2%F0%9D%90%A7%F0%9D%90%A7%F0%9D%90%9E%F0%9D%90%AB%F0%9D%90%AC-activity-7258071749244665856-TGh7)


## How-To's
Download dataset files:
```
python -m src.data.download
```

Run single model training using parameters from `model_config.json`:
```
python -m src.model.training_single
```

Run hyperparameter optimization with `optuna`:
```
python -m src.model.training_optuna
```

Evaluate resulting model and send to submission API with:
```
python -m src.model.evaluate --run_id 6aeebe13-4ca0-456f-9476-a956f738e598 --submit
```

# To-Do List
List of to-do's, feel free to add new ones, or pick any of these up.
- Implement config options for data processing
- Implement hyperparameter optimization
- Implement option to cast `PRICE` to `log_price` (stats trick)
- Implement option to normalize numeric features
- Implement option to reduce dimensionality of high dimension categorical features (`BUILDING_ID`, `UNIT_ID`, `STREET_CODE`)
- Investigate better imputation methods


## Other Resources
- Danmarks statistik: https://www.dst.dk/en/Statistik/dokumentation/nomenklaturer/amt-kom
