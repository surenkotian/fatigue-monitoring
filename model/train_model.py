import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

data = pd.read_csv("wfh_fatigue_data.csv")
x = data[["EAR", "MAR"]]
y = data["State"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(x_train, y_train)

joblib.dump(model, "fatigue_model.pkl")
preds = model.predict(x_test)
print(classification_report(y_test, preds))