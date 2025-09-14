# Wafer Map Defect Classifier (Streamlit + GitHub)

A lightweight, hands-on demo for semiconductor yield engineering. It classifies synthetic wafer map patterns into five classes: `center`, `edge_ring`, `scratch`, `donut`, and `random`. The project is designed to be GitHub- and Streamlit-ready with small artifacts (<25 MB).

## Quick start
```bash
pip install -r requirements.txt
streamlit run app/app.py
```

## Project layout
```
wafer_map_defect_classifier/
├─ app/
│  ├─ app.py
│  ├─ pages/
│  │  ├─ 1_📈_Dashboard.py
│  │  └─ 2_🔎_Model_Analysis.py
│  ├─ components/
│  │  └─ wafer_plot.py
├─ data/
│  └─ sample/
│     └─ wafer_samples.csv
├─ models/
│  ├─ trained/
│  │  └─ model.pkl
│  └─ train_notebooks/
├─ src/
│  ├─ generate_data.py
│  ├─ features.py
│  ├─ train.py
│  └─ predict.py
├─ tests/
│  └─ test_smoke.py
├─ .streamlit/config.toml
├─ requirements.txt
├─ README.md
└─ LICENSE
```

## Re-train locally (optional)
```bash
python -m src.train --n 600 --seed 7 --save models/trained/model.pkl --csv data/sample/wafer_samples.csv
```