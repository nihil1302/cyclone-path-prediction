🌀 Cyclone Path Prediction

A B.Tech Project that predicts the next position of a tropical cyclone using historical track data from **NOAA IBTrACS** and **XGBoost regression**.

How to Run

1. Clone the repo
```bash
git clone https://github.com/nihil1302/cyclone-path-prediction.git
cd cyclone-path-prediction
```
2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Place the IBTrACS CSV in the same folder as `streamlit_app.py`
> Download from: https://www.ncei.noaa.gov/products/international-best-track-archive

4. Launch the app
```bash
streamlit run streamlit_app.py
```
> The model trains automatically on first run (~2 min). No extra steps needed.

Tech Stack
`Python` · `XGBoost` · `Pandas` · `Scikit-learn` · `Matplotlib` · `Streamlit`

Model Accuracy
| | Latitude | Longitude |
|--|--|--|
| RMSE | 0.3679° (~41 km) | 0.8411° (~93 km) |
