# 🔍Fraud-Detection System
An end-to-end Fraud Detection System built with Machine Learning and a Streamlit Dashboard.
This project detects fraudulent transactions in financial data, visualizes patterns, and allows exporting fraud cases for further investigation.


# 🚀Features

📂 Upload transaction CSV files

🤖 Predict fraud vs. non-fraud using a trained ML model

📊 Interactive visualizations:

  1) Fraud vs. Non-fraud counts

  2) Transaction amount distribution

  3) Fraud trends over time

⚠️ Fraud alerts highlighted in red

📥 Export detected fraud transactions as a CSV

# 🛠️ Tech Stack

**Python 3.10+**

**Libraries:**
 
- **scikit-learn** → ML model training & evaluation  
- **joblib** → Model saving/loading  
- **pandas, numpy** → Data handling  
- **matplotlib, seaborn** → Visualization  
- **streamlit** → Dashboard UI

# 🔄 How It Works

**1) Data Preprocessing & Evaluation:**

 - Cleaned and transformed transaction dataset.
 - Trained multiple models on the data.
 - Evaluated using Accuracy, Precision, Recall, and F1-score for balanced fraud detection performance.

**2) Model Training:**

  - Logistic Regression & Random Forest models tested.
  - Final model chosen based on evaluation metrics.
  - Best model saved with joblib.


**3) Dashboard:**

 - Upload new CSV transactions
 - Model classifies each transaction (Fraud = 1, Non-Fraud = 0)
 - Results displayed in tables + graphs.
 - Fraud transactions can be downloaded as CSV for further use.


 ## 📁 Project Structure

 ```

fraud detection/
│
├── dashboard2.py # Main Streamlit app
├── preprocesing.py # Data fetching & preprocessing
├── model_training.py # Training the both models and choosing the best
├── cleaned_creditcard.csv # Cleaned dataset after preprocessing 
├── creditcard.csv # Dataset choosen
├── fraud_detection_model.pkl # Best Model saved after training
├── requirements.txt # Dependencies
├── visuals.ipynb # Feature Engineering with data analysis
├── README.md # Project documentation
└── detection/ # Virtual environment




```

---


### ⚙️Installation & Setup
**1) Clone the repository:**
 
 ```bash
 git clone https://github.com/vineethdhagey/Fraud-Detection-.git
 cd Fraud-Detection-
```

**2) Create and activate a virtual environment**

   ```bash
    Windows:
   python -m venv venv

   venv\Scripts\activate
   ```
**3) Install dependencies**

   ```bash
   pip install -r requirements.txt

```

**4) Run the app:**
 ```bash
streamlit dashbaord2.py

```

# 🖼️ Screenshots

<img width="1783" height="604" alt="Screenshot 2025-09-01 102744" src="https://github.com/user-attachments/assets/1f971db9-c70e-411b-8ea2-cc4c23cf0174" />


<img width="1769" height="760" alt="Screenshot 2025-09-01 102811" src="https://github.com/user-attachments/assets/844460fa-f877-45fa-a1ec-bc10cc11f42a" />

<img width="1758" height="731" alt="Screenshot 2025-09-01 102832" src="https://github.com/user-attachments/assets/6caca0a9-efc3-4bb4-a918-1a43668c96db" />


# 🤝 Contributing

Contributions are welcome!

1) Fork the repo

2) Create a new branch

3) Submit a Pull Request

# 📄 License
This project is licensed under the MIT License.



