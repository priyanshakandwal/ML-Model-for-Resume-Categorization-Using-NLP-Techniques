# ML Model for Resume Categorization Using NLP Techniques
 EDA on resume data with Pandas/Seaborn/Numpy to analyze job categories for future ML classification.
📄 Resume Classification using Machine Learning
A machine learning project that classifies resumes into various job categories (e.g., Data Science, HR, etc.) based on their content. This is useful for HR automation and resume screening. The project uses text preprocessing, NLP, and classification techniques.

📁 Dataset
UpdatedResumeDataSet.csv: Contains resumes with a Category column representing the job domain.

📌 Features:
Resume: Text content of the resume.

Category: Target label for classification.

📊 Exploratory Data Analysis
Performed basic EDA on the dataset:

.head(), .tail(), .describe(), .nunique() for data exploration.

Visualized distribution of categories using:

Countplot

Pie chart

🧼 Data Cleaning
Custom function cleanResume() used for:

Removing URLs, mentions, hashtags

Removing punctuation, symbols, and special characters

Lowercasing and whitespace normalization

Also used NLTK for:

Stopword removal

Tokenization

python
Copy
Edit
def cleanResume(txt):
    cleanText = re.sub('http\\S+\\s', ' ', txt)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\\S+\\s', ' ', cleanText)
    cleanText = re.sub('@\\S+', ' ', cleanText)
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub('\s+', ' ', cleanText)
    return cleanText
🤖 Model Training
Used machine learning to classify resumes into categories.

Steps:
Data Cleaning

Text Vectorization (likely TfidfVectorizer or similar)

Model training using classifiers like:

K-Nearest Neighbors (KNN)

(Other models if included)

Model Saving:
Used joblib to save the trained model for future inference:

python
Copy
Edit
import joblib
joblib.dump(knn_model, 'knn_model.pkl')
🧠 Libraries Used
pandas

matplotlib, seaborn

nltk

re

sklearn

joblib

🛠 To Run:
Clone the repo or upload the notebook to Google Colab.

Make sure the dataset path is correct:
/content/drive/MyDrive/priyansha ml/UpdatedResumeDataSet.csv

Install necessary libraries (Colab-friendly).

Run the cells step-by-step.

📄 License
This project is licensed under the MIT License.
