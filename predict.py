import joblib
import pandas as pd
model=joblib.load("telecom_tower_model.pkl")
new_data=pd.DataFrame({
'Temperature_C':[55],
'Battery_Voltage':[46.3],
'Power_Consumption_W':[1000],
'Signal_Strength_Percent':[88],
'Fan_Speed_RPM':[4000],
'Humidity_Percent':[50],
'Traffic_Load':[2000],
'Tower_Age_Years':[14]
})
prediction=model.predict(new_data)
if prediction[0]==1:
 print("Hardware Failure Predicted")
else:
 print("Tower is Healthy")
