import base64
from flask import Flask, request, render_template
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import pandas as pd
import io
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)


@app.route('/home.html')
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about.html')
def about():
    return render_template('about.html')


@app.route('/get_patient_details.html')
def get_patient_details():
    return render_template('get_patient_details.html')


@app.route('/sumbit_patient_sym', methods=['GET ', 'POST'])
def predict():
    data = pd.read_csv("breast_cancer.csv")
    data = data.drop(['cohort', 'cancer_type'], axis=1)
    new_data = data.copy()
    death_binary_map = {'Living': 0, 'Died of Disease': 1}
    er_status_binary_map = {'Positive': 1, 'Negative': 0}
    cellularity_binary_map = {'High': 1, 'Moderate': 0.5, 'low': 0}
    pr_status_binary_map = {'Positive': 1, 'Negative': 0}
    her2_status_binary_map = {'Positive': 1, 'Negative': 0}
    type_of_breast_surgery_binary_map = {'BREAST CONSERVING': 1, 'MASTECTOMY': 0}
    cancer_type_detailed_binary_map = {'Breast Invasive Ductal Carcinoma': 1, 'Breast Invasive Lobular Carcinoma': 0.5,
                                       'Breast Mixed Ductal and Lobular Carcinoma': 0}
    inferred_menopausal_state_binary_map = {'Post': 1, 'Pre': 0}
    primary_tumor_laterality_binary_map = {'Left': 1, 'Right': 0}
    new_data['death_binary'] = new_data['death_from_cancer'].map(death_binary_map)
    new_data['er_status_binary'] = new_data['er_status'].map(er_status_binary_map)
    new_data['cellularity_binary'] = new_data['cellularity'].map(cellularity_binary_map)
    new_data['pr_status_binary'] = new_data['pr_status'].map(pr_status_binary_map)
    new_data['her2_status_binary'] = new_data['her2_status'].map(her2_status_binary_map)
    new_data['type_of_breast_surgery_binary'] = new_data['type_of_breast_surgery'].map(
        type_of_breast_surgery_binary_map)
    new_data['cancer_type_detailed_binary'] = new_data['cancer_type_detailed'].map(cancer_type_detailed_binary_map)
    new_data['inferred_menopausal_state_binary'] = new_data['inferred_menopausal_state'].map(
        inferred_menopausal_state_binary_map)
    new_data['primary_tumor_laterality_binary'] = new_data['primary_tumor_laterality'].map(
        primary_tumor_laterality_binary_map)
    binary_encoding = pd.get_dummies(new_data['er_status_measured_by_ihc'], drop_first=True, dummy_na=False)
    new_data = new_data.drop(
        ['cellularity', 'er_status', 'her2_status', 'death_from_cancer', 'pr_status', 'er_status_measured_by_ihc',
         'type_of_breast_surgery', 'cancer_type_detailed', 'inferred_menopausal_state', 'primary_tumor_laterality'],
        axis=1)
    new_data = pd.concat([new_data, binary_encoding], axis=1)
    new_data = new_data.dropna()
    features = ['age_at_diagnosis', 'chemotherapy', 'neoplasm_histologic_grade', 'hormone_therapy',
                'lymph_nodes_examined_positive', 'mutation_count', 'nottingham_prognostic_index', 'radio_therapy',
                'tumor_size', 'tumor_stage', 'er_status_binary', 'cellularity_binary', 'pr_status_binary',
                'her2_status_binary', 'type_of_breast_surgery_binary', 'cancer_type_detailed_binary',
                'inferred_menopausal_state_binary', 'primary_tumor_laterality_binary']
    X = new_data[features]
    y_death = new_data['death_binary']
    y_survival = new_data['overall_survival_months']
    rf_classifier_death = RandomForestClassifier(random_state=42)
    rf_classifier_death.fit(X, y_death)
    rf_regressor_survival = RandomForestRegressor(random_state=42)
    rf_regressor_survival.fit(X, y_survival)
    age_at_diagnosis = float(request.form.get('age_at_diagnosis'))
    chemotherapy = int(request.form.get('chemotherapy'))
    neoplasm_histologic_grade = int(request.form.get('neoplasm_histologic_grade'))
    hormone_therapy = int(request.form.get('hormone_therapy'))
    lymph_nodes_examined_positive = int(request.form.get('lymph_nodes_examined_positive'))
    mutation_count = int(request.form.get('mutation_count'))
    nottingham_prognostic_index = float(request.form.get('nottingham_prognostic_index'))
    radio_therapy = int(request.form.get('radio_therapy'))
    tumor_size = float(request.form.get('tumor_size'))
    tumor_stage = int(request.form.get('tumor_stage'))
    er_status_binary = er_status_binary_map.get(request.form.get('er_status'))
    cellularity_binary = cellularity_binary_map.get(request.form.get('cellularity'))
    pr_status_binary = pr_status_binary_map.get(request.form.get('pr_status'))
    her2_status_binary = her2_status_binary_map.get(request.form.get('her2_status'))
    type_of_breast_surgery_binary = type_of_breast_surgery_binary_map.get(request.form.get('type_of_breast_surgery'))
    cancer_type_detailed_binary = cancer_type_detailed_binary_map.get(request.form.get('cancer_type_detailed'))
    inferred_menopausal_state_binary = inferred_menopausal_state_binary_map.get(
        request.form.get('inferred_menopausal_state'))
    primary_tumor_laterality_binary = primary_tumor_laterality_binary_map.get(
        request.form.get('primary_tumor_laterality'))
    user_input = pd.DataFrame({
        'age_at_diagnosis': [age_at_diagnosis],
        'chemotherapy': [chemotherapy],
        'neoplasm_histologic_grade': [neoplasm_histologic_grade],
        'hormone_therapy': [hormone_therapy],
        'lymph_nodes_examined_positive': [lymph_nodes_examined_positive],
        'mutation_count': [mutation_count],
        'nottingham_prognostic_index': [nottingham_prognostic_index],
        'radio_therapy': [radio_therapy],
        'tumor_size': [tumor_size],
        'tumor_stage': [tumor_stage],
        'er_status_binary': [er_status_binary],
        'cellularity_binary': [cellularity_binary],
        'pr_status_binary': [pr_status_binary],
        'her2_status_binary': [her2_status_binary],
        'type_of_breast_surgery_binary': [type_of_breast_surgery_binary],
        'cancer_type_detailed_binary': [cancer_type_detailed_binary],
        'inferred_menopausal_state_binary': [inferred_menopausal_state_binary],
        'primary_tumor_laterality_binary': [primary_tumor_laterality_binary],
    })
    predicted_death = rf_classifier_death.predict(user_input)[0]
    predicted_survival = rf_regressor_survival.predict(user_input)[0]
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
    axes[0].bar(['Living', 'Died of Disease'], [1 - predicted_death, predicted_death], color=['green', 'red'])
    axes[0].set_title('Death Prediction')
    axes[1].bar(['Survival Months'], [predicted_survival], color='blue')
    axes[1].set_title('Survival Months Prediction')
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    img_base64 = base64.b64encode(img.getvalue()).decode()
    return render_template('graph.html', img_base64=img_base64)


if __name__ == '__main__':
    app.run(debug=True)
