# Patient Outcome Prediction (Breast Cancer Survival Analysis)

A **Flask-based Machine Learning web application** that predicts **breast cancer patient outcomes** — whether a patient is likely to survive or not — and estimates **survival months** using a **Random Forest model** trained on clinical data.

This project combines **data preprocessing**, **ML model training**, and a **web-based visualization** to assist in healthcare data-driven decisions.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Input Parameters](#input-parameters)
- [Technologies Used](#technologies-used)
- [Notes](#notes)

---

## Overview

The **Patient Outcome Prediction System** uses clinical data to:

- Predict whether a patient is likely to **survive or die from breast cancer**  
- Estimate the **expected survival duration (in months)**  
- Display **visual graphs** showing predictions (death probability and survival months)

The model is built using **RandomForestClassifier** and **RandomForestRegressor** from `scikit-learn`, and deployed through a **Flask web interface**.

---

## Features

-  Predicts **patient survival** and **disease outcome**
-  Estimates **overall survival time (months)**
-  Trains **Random Forest models** dynamically during runtime
-  Encodes and preprocesses multiple categorical clinical features
-  Accepts form-based input through Flask templates
-  Generates visual graphs using **Matplotlib**
-  User-friendly interface built with HTML templates

---

## Project Structure

```
Patient_Outcome_Prediction/
│
├─ static/                     # Static files (CSS, images, JS)
│   └─ css/                    # Styling for templates
│
├─ templates/                  # Flask HTML templates
│   ├─ home.html
│   ├─ about.html
│   ├─ get_patient_details.html
│   └─ graph.html              # Displays prediction results and charts
│
├─ app.py                      # Main Flask application
├─ breast_cancer.csv           # Dataset used for training (must be present)
└─ requirements.txt            # Python dependencies
```

---

##  Installation

1. **Clone the repository**

```bash
git clone https://github.com/NGowthamKumar/Patient_Outcome_Prediction
cd Patient_Outcome_Prediction
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Ensure the dataset is available**

Place `breast_cancer.csv` in the root folder.

4. **Run the Flask app**

```bash
python app.py
```

5. **Access the web app**

Open your browser and visit:  
`http://127.0.0.1:5000/`

---

## Usage

1. Go to the **Home Page** (`/home.html`)  
2. Navigate to **Get Patient Details**  
3. Enter clinical parameters such as:
   - Age at diagnosis  
   - Chemotherapy (0 or 1)  
   - Tumor size, stage, grade, etc.  
4. Submit the form  
5. View prediction graphs showing:
   - Probability of survival vs. death  
   - Predicted survival duration in months  

---

## Input Parameters

| Feature | Description |
|----------|-------------|
| `age_at_diagnosis` | Age when the patient was diagnosed |
| `chemotherapy` | Whether the patient received chemotherapy (0/1) |
| `neoplasm_histologic_grade` | Tumor grade |
| `hormone_therapy` | Whether hormone therapy was given (0/1) |
| `lymph_nodes_examined_positive` | Number of positive lymph nodes |
| `mutation_count` | Total genetic mutations detected |
| `nottingham_prognostic_index` | Prognostic index score |
| `radio_therapy` | Whether radiation therapy was given (0/1) |
| `tumor_size` | Tumor size (in mm) |
| `tumor_stage` | Cancer stage (1–4) |
| `er_status`, `pr_status`, `her2_status` | Receptor statuses (Positive/Negative) |
| `type_of_breast_surgery` | Type of surgery (Mastectomy / Breast Conserving) |
| `inferred_menopausal_state` | Menopausal status (Pre/Post) |
| `primary_tumor_laterality` | Tumor side (Left/Right) |

---

## Technologies Used

- **Python**  
- **Flask**  
- **Pandas**  
- **Scikit-learn**  
- **Matplotlib**  
- **HTML/CSS** for UI templates  

---

## Machine Learning Models Used

- `RandomForestClassifier` → Predicts **death_from_cancer** (Living / Died)  
- `RandomForestRegressor` → Predicts **overall_survival_months**  

---

## Notes

- Ensure **`breast_cancer.csv`** is correctly formatted and available.  
- Flask app runs in **debug mode** by default.  
- You can extend this project by saving trained models using `joblib` for faster predictions.  
- Default port: **5000**

---

