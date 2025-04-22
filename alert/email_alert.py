import yagmail
import pandas as pd

data = pd.read_csv("wfh_fatigue_data.csv")
drowsy_count = data[data['State'] == 'Drowsy'].shape[0]

if drowsy_count > 10:
    yag = yagmail.SMTP("surenkotian4@gmail.com", "mumma2005")
    yag.send(
        to="surenkotian10@gmail.com",
        subject="🚨 Fatigue Alert: Remote Employee Detected Drowsy",
        contents="More than 10 drowsy moments detected. Check the dashboard for detailed report."
    )   