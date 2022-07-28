import pickle
import pandas as pd

model = pickle.load(open('grid.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
class_names = [1,2,3]

def predict(df):
    df = df[['baseline value','accelerations','fetal_movement','uterine_contractions','light_decelerations','severe_decelerations',
             'prolongued_decelerations','abnormal_short_term_variability','mean_value_of_short_term_variability',
             'percentage_of_time_with_abnormal_long_term_variability','mean_value_of_long_term_variability','histogram_width',
             'histogram_min','histogram_max','histogram_number_of_peaks','histogram_number_of_zeroes','histogram_mode',
             'histogram_mean','histogram_median','histogram_variance','histogram_tendency']]
    df = pd.DataFrame(scaler.transform(df))
    numpy_array = df.to_numpy()
    predictions = model.predict(numpy_array)
    output = predictions.astype(int)
    output = output.tolist()
    return output