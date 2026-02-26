from flask import Flask, render_template, request, send_file
import os
import requests
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime, timedelta
import warnings
import numpy as np

from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA

from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from openpyxl import Workbook

# PDF Imports
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

load_dotenv()
API_KEY = "6b6e04f73541e221e19bfe92764a6936"

app = Flask(__name__)

# GLOBALS for export
predicted_temp = []
predicted_humidity = []
mae_temp = mae_hum = rmse_temp = rmse_hum = mape_temp = mape_hum = 0
irrigation_msg = ""
city_selected = ""


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        city = request.form.get("city", "").strip()

        if not city:
            return render_template("404_error.html", error_message="Enter a valid city name")

        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
        resp = requests.get(url).json()

        if str(resp.get("cod")) != "200":
            return render_template("404_error.html", error_message=resp.get("message"))

        return render_template("index.html",
                               status=True,
                               city=resp["name"],
                               country=resp["sys"]["country"],
                               current_temp=round(resp["main"]["temp"]),
                               feels_like=round(resp["main"]["feels_like"], 1),
                               temp_min=round(resp["main"]["temp_min"], 1),
                               temp_max=round(resp["main"]["temp_max"], 1),
                               humidity=round(resp["main"]["humidity"]),
                               description=resp["weather"][0]["description"]
                               )
    return render_template("index.html", status=False)


@app.route("/predict-weather", methods=["POST"])
def prediction():
    global predicted_temp, predicted_humidity
    global mae_temp, mae_hum, rmse_temp, rmse_hum, mape_temp, mape_hum
    global irrigation_msg, city_selected

    city = request.form.get("city", "").strip()
    city_selected = city

    url = f"https://api.openweathermap.org/data/2.5/forecast?q={city}&appid={API_KEY}&units=metric"
    data = requests.get(url).json()

    if str(data.get("cod")) != "200":
        return render_template("404_error.html", error_message=data.get("message"))

    entries = data["list"][:48]

    hours, temps, hums = [], [], []
    for e in entries:
        hours.append(e["dt_txt"])
        temps.append(e["main"]["temp"])
        hums.append(e["main"]["humidity"])

    df = pd.DataFrame({"hours": hours, "temp": temps, "hum": hums})

    warnings.filterwarnings("ignore")

    temp_series = df["temp"]
    hum_series = df["hum"]

    order_t = auto_arima(temp_series).get_params()["order"]
    order_h = auto_arima(hum_series).get_params()["order"]

    model_t = ARIMA(temp_series, order=order_t).fit()
    model_h = ARIMA(hum_series, order=order_h).fit()

    start = len(temp_series)

    predicted_temp = list(model_t.predict(start, start + 4))
    predicted_humidity = list(model_h.predict(start, start + 4))

    # Performance Matrix
    actual_temp = list(temp_series[-5:])
    actual_hum = list(hum_series[-5:])

    mae_temp = mean_absolute_error(actual_temp, predicted_temp)
    mse_temp = mean_squared_error(actual_temp, predicted_temp)
    rmse_temp = sqrt(mse_temp)
    mape_temp = np.mean(abs((np.array(actual_temp)-np.array(predicted_temp))/np.array(actual_temp))) * 100

    mae_hum = mean_absolute_error(actual_hum, predicted_humidity)
    mse_hum = mean_squared_error(actual_hum, predicted_humidity)
    rmse_hum = sqrt(mse_hum)
    mape_hum = np.mean(abs((np.array(actual_hum)-np.array(predicted_humidity))/np.array(actual_hum))) * 100

    labels = [(datetime.now() + timedelta(hours=3*(i+1))).strftime("%H:%M") for i in range(5)]

    # WRITE IRRIGATION MESSAGE
    if predicted_humidity[0] >= 75:
        irrigation_msg = "⚠ Rain expected within 1 hour. Do NOT irrigate."
    elif predicted_humidity[1] >= 75:
        irrigation_msg = "⚠ Rain expected within 2 hours."
    elif predicted_humidity[2] >= 75:
        irrigation_msg = "⚠ Rain expected within 3 hours."
    else:
        irrigation_msg = "✔ Safe to irrigate. No rain expected soon."

    return render_template("index.html",
                           predict_status=True,
                           city=city,
                           tlabels=labels,
                           hlabels=labels,
                           tvalues=[round(x, 2) for x in predicted_temp],
                           hvalues=[round(x, 2) for x in predicted_humidity],
                           temperature_1=round(predicted_temp[0], 1),
                           temperature_2=round(predicted_temp[1], 1),
                           temperature_3=round(predicted_temp[2], 1),
                           temperature_4=round(predicted_temp[3], 1),
                           temperature_5=round(predicted_temp[4], 1),
                           humidity_1=round(predicted_humidity[0], 1),
                           humidity_2=round(predicted_humidity[1], 1),
                           humidity_3=round(predicted_humidity[2], 1),
                           humidity_4=round(predicted_humidity[3], 1),
                           humidity_5=round(predicted_humidity[4], 1),
                           mae_temp=round(mae_temp, 3),
                           mse_temp=round(mse_temp, 3),
                           rmse_temp=round(rmse_temp, 3),
                           mape_temp=round(mape_temp, 2),
                           mae_hum=round(mae_hum, 3),
                           mse_hum=round(mse_hum, 3),
                           rmse_hum=round(rmse_hum, 3),
                           mape_hum=round(mape_hum, 2),
                           irrigation_msg=irrigation_msg,
                           )


@app.route("/download_excel")
def download_excel():

    wb = Workbook()
    ws = wb.active
    ws.append(["Hour", "Temp", "Humidity"])
    for i in range(5):
        ws.append([f"Hour {i+1}", predicted_temp[i], predicted_humidity[i]])

    ws.append([])
    ws.append(["Metric", "Temp", "Humidity"])
    ws.append(["MAE", mae_temp, mae_hum])
    ws.append(["RMSE", rmse_temp, rmse_hum])
    ws.append(["MAPE%", mape_temp, mape_hum])

    filename = "Prediction_Report.xlsx"
    wb.save(filename)
    return send_file(filename, as_attachment=True)


@app.route("/download_pdf")
def download_pdf():

    filename = "Weather_Report.pdf"
    doc = SimpleDocTemplate(filename)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("<b>🌤 WEATHER PREDICTION REPORT</b>", styles["Title"]))
    story.append(Paragraph(f"<b>City:</b> {city_selected}", styles["Normal"]))
    story.append(Spacer(1, 15))

    # Prediction table
    table_data = [["Hour", "Temp (°C)", "Humidity (%)"]]
    for i in range(5):
        table_data.append([f"Hour {i+1}", predicted_temp[i], predicted_humidity[i]])

    table = Table(table_data)
    table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 1, colors.black)]))
    story.append(table)
    story.append(Spacer(1, 15))

    # PERFORMANCE TABLE
    perf = [["Metric", "Temp", "Humidity"],
            ["MAE", mae_temp, mae_hum],
            ["RMSE", rmse_temp, rmse_hum],
            ["MAPE %", mape_temp, mape_hum]]
    perf_tbl = Table(perf)
    perf_tbl.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 1, colors.red)]))
    story.append(perf_tbl)

    story.append(Spacer(1, 15))
    story.append(Paragraph(f"<b>💧 Irrigation Advice:</b> {irrigation_msg}", styles["Normal"]))

    # SAVE PDF
    doc.build(story)

    return send_file(filename, as_attachment=True)


@app.errorhandler(404)
def page_not_found(e):
    return render_template("404_error.html", error_message="Page Not Found"), 404


if __name__ == "__main__":
    app.run(debug=True)
