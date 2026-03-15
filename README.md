![Banner](https://github.com/user-attachments/assets/04f47400-1087-48b3-9679-4046b7df8cd3)

  <h1 align="center">🧠NeuroRest</h1>
  
> *AI-powered Sleep Quality & Stress Predictor*

<p align="center">
  <img src="https://img.shields.io/badge/Status-Active-success?style=flat-square" />
  <img src="https://img.shields.io/badge/License-MIT-blue?style=flat-square" />
  <img src="https://img.shields.io/github/stars/krRaviongit/NeuroRest?style=social" />
  <img src="https://img.shields.io/badge/Python-3.10%2B-yellow?style=flat-square" />
</p>

---

## ✨ Overview

**NeuroRest** is a machine learning project that predicts **sleep quality** and **stress levels** based on lifestyle factors like:

* 💤 Sleep Hours
* 🚶 Physical Activity
* 📱 Screen Time
* 👣 Steps
* 👤 Age, Gender, Occupation

The project leverages data-driven insights to help individuals **improve their well-being** by identifying patterns in daily habits.

---

## 🎯 Features

✅ Predict **sleep quality** (good, average, poor)
✅ Predict **stress levels** (low, medium, high)
✅ Clean & beginner-friendly codebase
✅ Interactive visualizations for insights
✅ Ready for deployment on Streamlit / Flask

---

## 🛠️ Tech Stack

* **Languages**: Python 🐍
* **Libraries**:

  * `pandas`, `numpy` → Data Processing
  * `scikit-learn` → Machine Learning Models
  * `matplotlib`, `seaborn` → Visualizations
  * `streamlit` → Deployment

---

## 📊 Dataset  

This project uses two Kaggle datasets that provide information on **sleep quality** and **stress factors**:  

1. **Sleep Health and Lifestyle Dataset**  
   - File: `Sleep_health_and_lifestyle_dataset.csv`  
   - Source: [Kaggle - Sleep Health and Lifestyle Dataset](https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset)  
   - Contains information about individuals' **sleep patterns and lifestyle habits**.  
   - **Key Features:**  
     - Age, Gender, Occupation  
     - Sleep Duration (hours)  
     - Quality of Sleep (scale 1–10)  
     - Physical Activity Level  
     - Stress Level (scale 1–10)  
     - BMI Category, Blood Pressure, Heart Rate, Daily Steps  

2. **Student Stress Factors Dataset**  
   - File: `Student Stress Factors.csv`  
   - Source: [Kaggle - Student Stress Factors](https://www.kaggle.com/datasets/samyakb/student-stress-factors)  
   - Focuses on **stress levels among students** based on academic, lifestyle, and social factors.  
   - **Key Features:**  
     - Sleep Duration, Screen Time  
     - Academic Pressure, Study Hours  
     - Diet Quality, Physical Activity  
     - Stress Levels (categorized)  

🔗 These datasets are analyzed together to build the **NeuroRest: Stress and Sleep Predictor**, highlighting how **lifestyle, academic, and health factors influence stress and sleep quality**.  

---

## 📂 Project Structure

```
NeuroRest/
│── data/                 # Datasets  
│── notebooks/            # Jupyter notebooks for EDA & training  
│── src/                  # Source code (models, preprocessing)  
│── app.py                # Streamlit app (frontend)  
│── requirements.txt      # Dependencies  
│── README.md             # Documentation  
```

---

## ⚡ Demo

👉 [Live Demo Link](https://neurorest.streamlit.app/)

---

## 📊 Sample Output
| 🧠 Data Insights | 📈 Model Results | 🔮 Predictions |
|:----------------:|:----------------:|:---------------:|
| ![EDA](https://github.com/user-attachments/assets/18dafba7-2bf5-4fe3-8158-7f2c344af8a6) | ![Model](https://github.com/user-attachments/assets/6e3eae5a-36ef-40fd-b647-24219fac57a4) | ![Pred](https://github.com/user-attachments/assets/b3bccbb7-6ddc-4270-8478-855281fc26ab) |


**Data Exploration Section**

![Image](https://github.com/user-attachments/assets/18dafba7-2bf5-4fe3-8158-7f2c344af8a6)


![Image](https://github.com/user-attachments/assets/173c0384-7bde-4d3e-aec1-c486bb422883)


![Image](https://github.com/user-attachments/assets/af894a90-4641-4da5-a61e-2885047880cd)


![Image](https://github.com/user-attachments/assets/802077e0-a5a1-43c1-9089-49cefa912b66)

---

**Model Training Results Section**

![Image](https://github.com/user-attachments/assets/6e3eae5a-36ef-40fd-b647-24219fac57a4)

---

 **Predictions Section**

![Image](https://github.com/user-attachments/assets/b3bccbb7-6ddc-4270-8478-855281fc26ab)


---

## 🚀 Getting Started

### 1️⃣ Clone the repo

```bash
git clone https://github.com/your-username/NeuroRest.git
cd NeuroRest
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the app

```bash
streamlit run app.py
```

---

## 🌱 Future Enhancements

* Add personalized health recommendations
* Integrate wearable device data (Fitbit, Mi Band, Apple Watch)
* Deploy as a mobile app

---

## 👨🏻‍💻 Contributors

* **Kumar Ravi** – [GitHub](https://github.com/krRaviongit)
* **Abinash Giri** – [GitHub](https://github.com/ABIN2005)
* **Jay Gupta**
---

## 📜 License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.



---


<p align="center">Made with ❤️ by <b>Kumar Ravi</b></p>



