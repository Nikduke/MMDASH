# MMDASH – Overvoltage Dashboard

This is a Python Dash application that replicates the behavior of the Excel dashboard in `Overvoltages & Initial voltages.xlsx`. It visualizes overvoltage results from electromagnetic transient (EMT) simulations stored in `MM results.csv`.

The dashboard enables dynamic filtering by case, bus voltage, and bus name, and displays peak values and trends of key overvoltage metrics.

---

## ✅ Features

- Filter by:
  - ✅ Case name (multi-select)
  - ✅ Bus voltage [kV] (single-select)
  - ✅ Bus name (filtered based on voltage)

- Fixed X-axis:
  - All plots use `Run#` as the time/evolution axis

- Dynamic KPI cards:
  - Max LGp [pu]
  - Max LLp [pu]
  - Max LGr [pu]
  - Max LLr [pu]
  - Max TOV duration [s]

- Graphs:
  - Instantaneous LGp vs. Run#
  - Instantaneous LLp vs. Run#
  - RMS rise LGr vs. Run#
  - RMS rise LLr vs. Run#
  - TOV duration vs. Run#

---

## 🗂 File Structure

```
MMDASH/
├── dashboard.py         # Dash app layout and callbacks
├── data_model.py        # Data filtering and pivot logic
├── MM results.csv       # Raw input data (simulation results)
├── requirements.txt     # Python package requirements
├── test_data_model.py   # Optional unit tests
```

---

## 🚀 Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the dashboard

```bash
python dashboard.py
```

Then visit [http://127.0.0.1:8050](http://127.0.0.1:8050) in your browser.

---

## ⚙️ Architecture

- Built using [Dash](https://dash.plotly.com/) and [Plotly](https://plotly.com/python/)
- Data loaded from CSV using `pandas`
- KPI and graph logic separated into `data_model.py` for modularity
- All UI interactivity handled in `dashboard.py` using Dash callbacks

---

## 📌 Limitations (Original Version)

- Only one voltage level can be selected at a time
- Only one bus name can be selected at a time (manually)
- X-axis is fixed to `Run#`

---

## 📈 Future Enhancements

- Multi-voltage filtering with auto-detected buses
- X-axis toggle: `Run#` ↔ `Tswitch_a [s]`
- Export filtered view as CSV or image
- KPI trend over time

---

## 👤 Author

Developed by [Your Name], 2025  
MIT License