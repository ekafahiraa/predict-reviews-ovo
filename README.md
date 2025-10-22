🧠 Predict Reviews OVO

Aspect-Based Sentiment Analysis on User Perceptions of the OVO App Using Support Vector Machine (SVM)
This project is a Flask-based web application that performs sentiment analysis on OVO app reviews. It allows users to predict sentiment either by entering text or uploading a CSV file. The system visualizes sentiment distribution using charts and provides CSV download functionality for the predicted results.

🚀 Features
📝 Text Input Prediction: Users can predict sentiment by typing a review directly into the form.
📂 CSV File Upload: Users can upload a CSV file containing multiple reviews for bulk prediction.
👀 CSV Data Preview: Uploaded CSV data and prediction results are displayed in a clean, paginated table (10 rows per page).
📊 Data Visualization: Displays a bar chart and pie chart showing the sentiment distribution and percentage across different aspects.
💾 CSV Download: Users can download the CSV file containing the predicted sentiment results.

📊 Example Visualizations
Below are sample results from the prediction output:
- Bar Chart — Sentiment Distribution per Aspect
Displays the total count of positive and negative sentiments for each aspect.
- Pie Chart — Sentiment Percentage
Visualizes the proportion of each sentiment type across the dataset.

🧩 Tech Stack
- Python 3.12
- Flask
- NLTK
- Scikit-learn
- Matplotlib
- Pandas
- NumPy

⚙️ How to Run the Project
1️⃣ Clone the Repository
git clone https://github.com/ekafahiraa/predict-reviews-ovo.git
cd predict-reviews-ovo
2️⃣ Install Dependencies
Make sure you have pip and Python 3.12 installed.
pip install -r requirements.txt
3️⃣ Run the Flask App
python app.py
4️⃣ Open in Browser
http://127.0.0.1:5000/

💡 Notes
If running for the first time, download NLTK data:
import nltk
nltk.download('punkt')
nltk.download('stopwords')
The file model.pkl must contain a pre-trained SVM model.

📜 License
This project is open-source and intended for educational and research purposes.
