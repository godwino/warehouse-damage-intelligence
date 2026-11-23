# Warehouse Damage Intelligence Dashboard

## Business Problem
Warehouse damage creates avoidable cost, rework, claims, inventory inaccuracy, and service risk across grocery distribution operations. Leaders need a clear way to understand where damage happens, what process conditions drive it, and which corrective actions will have the strongest impact.

## Project Objective
This project delivers a portfolio-ready Streamlit dashboard and machine learning MVP that helps warehouse teams identify damage hotspots, quantify cost exposure, and prioritize process improvement actions.

## Why This Project Matters
- Product damage reduces fill rate and on-shelf availability.
- Damage drives write-offs, claims, and extra labor.
- Repeated damage patterns can point to slotting, congestion, equipment, and pallet quality problems.
- Data-driven visibility helps managers improve safety and productivity while protecting inventory.

## Ethical Design Note
This project focuses on process improvement and operational risk. It avoids using individual employee identity or picker experience as a prediction feature because damage risk can be affected by many system factors such as congestion, slotting, pallet stability, equipment condition, workload, and product characteristics.

## Dataset Overview
The synthetic dataset contains 5,000 warehouse damage records and includes:
- Operational context such as process area, handling stage, aisle, shift, and slot level
- Product attributes such as SKU, category, weight, cube, and cost per case
- Risk indicators such as congestion, pallet stability, equipment condition, and slotting risk
- Damage outcomes such as damaged cases, damage rate, damage cost, and high-damage classification

## Dashboard Features
- Executive KPI summary
- Damage hotspot analysis by SKU, aisle, trend, and damage type
- Process area analysis for cost, rate, and damaged volume
- SKU and category views including slot risk heatmaps
- Equipment and damage source analysis
- Operational risk analysis centered on environment and process conditions
- Prediction tool for high-damage risk scenarios
- Automated recommendations table
- Business impact savings scenarios

## Machine Learning Approach
The app trains a `RandomForestClassifier` to predict whether a record is high damage (`high_damage = 1`). The pipeline uses:
- `train_test_split`
- `ColumnTransformer`
- `OneHotEncoder`
- `Pipeline`
- `RandomForestClassifier`

`damage_source` is excluded from training and the prediction form because it is recorded after an incident. It remains available for historical analysis and corrective actions. Prediction-time recommendations use only entered operational conditions. Other inputs must be recorded before handling; `cases_handled` represents planned volume when scoring.

The model reports accuracy, precision, recall, F1 score, confusion matrix, classification report, and feature importance.

## Recommendation Engine
The recommendation engine converts high-risk operational patterns into action-focused suggestions, including:
- Dock safety and trailer communication reviews
- Pallet stability and wrapping audits
- Slotting and location-fit reviews
- Equipment inspection follow-up
- Receiving, putaway, replenishment, picking, and shipping SOP checks

## SQL Analytics Layer
The project also includes SQL Server scripts for:
- Table creation
- CSV import guidance
- KPI analysis
- Process area analysis
- SKU, category, and aisle analysis
- Damage source and equipment analysis
- High-priority action list
- Savings scenarios

## Business Impact
The dashboard estimates potential savings from reducing damage cost by 10%, 15%, and 20%. This helps connect operational improvements to financial outcomes.

## Project Structure
```text
warehouse-damage-intelligence/
|-- app.py
|-- generate_data.py
|-- requirements.txt
|-- README.md
|-- .gitignore
|-- data/
|   `-- warehouse_damage_data.csv
`-- sql/
    |-- create_warehouse_damage_table.sql
    |-- warehouse_damage_analysis.sql
    `-- README_SQL.md
```

## How to Run
```bash
pip install -r requirements.txt
python generate_data.py
streamlit run app.py
```

## Technologies Used
- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Plotly

## Future Improvements
- Connect to real WMS data
- Add Power BI version
- Add user login
- Add database storage
- Add weekly management report
- Add root cause investigation workflow
- Add before/after damage reduction tracking

## Regression Checks
Run `python -m unittest discover -s tests -v` to verify that training and prediction do not require damage source, changing it cannot change predictions, and historical source-based recommendations remain available.

The model uses synthetic data and a random holdout. These scores do not establish real-world prevention performance; validate input timing and use a time-based pilot evaluation before deployment.
