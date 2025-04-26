import joblib
 
model = joblib.load("model/gboost_model.joblib")
result_target = joblib.load("model/encoder_target.joblib")

def prediction(data):
    """Making prediction
 
    Args:
        data (Pandas DataFrame): Dataframe that contain all the preprocessed data
 
    Returns:
        str: Prediction result (Lulus, Putus Studi or Terdaftar)
    """
    result = model.predict(data)
    final_result = result_target.inverse_transform(result)[0]
    return final_result