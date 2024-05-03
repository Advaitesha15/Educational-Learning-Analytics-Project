import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
import random
import pandas as pd

# Load the synthetic data CSV file
data = pd.read_csv('D:\\synthetic_data.csv')

# Define input features and target variable
X = data[['Hours Studied', 'Attendance Percentage', 'Gender', 'Faculty Feedback']]
y = data['Previous Exam Score']

# Convert text data to numerical using TF-IDF
tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(X['Faculty Feedback'])
X_tfidf_df = pd.DataFrame(X_tfidf.toarray(), columns=tfidf.get_feature_names_out())
X = pd.concat([X[['Hours Studied', 'Attendance Percentage', 'Gender']], X_tfidf_df], axis=1)

# Convert categorical variables to numerical using one-hot encoding
X = pd.get_dummies(X, columns=['Gender'])

# Build a linear regression model
model = LinearRegression()
model.fit(X, y)

def predict_score():
    try:
        # Retrieve inputs from GUI fields
        hours_studied = float(hours_studied_entry.get())
        if not 0 <= hours_studied <= 24:
            raise ValueError("Hours studied must be between 0 and 24.")
        attendance_percentage = float(attendance_entry.get())
        if not 0 <= attendance_percentage <= 100:
            raise ValueError("Attendance percentage must be between 0 and 100.")
        faculty_feedback = faculty_feedback_entry.get()
        gender = gender_combobox.get()

        # Convert faculty feedback to TF-IDF
        faculty_feedback_tfidf = tfidf.transform([faculty_feedback])
        faculty_feedback_tfidf_df = pd.DataFrame(faculty_feedback_tfidf.toarray(), columns=tfidf.get_feature_names_out())

        # Create input for prediction
        input_data = {
            'Hours Studied': [hours_studied],
            'Attendance Percentage': [attendance_percentage],
            'Gender_Female': [1 if gender == 'Female' else 0],
            'Gender_Male': [1 if gender == 'Male' else 0]
        }
        input_data.update(faculty_feedback_tfidf_df.to_dict(orient='records')[0])

        # Make prediction using the model
        prediction = model.predict(pd.DataFrame(input_data))

        # Ensure predicted score is within valid range
        prediction = max(0, min(prediction, 100))  # Clip prediction to range [0, 100]

        # Display prediction
        messagebox.showinfo("Prediction", f"Predicted Exam Score: {prediction}")
    except ValueError as e:
        messagebox.showerror("Error", str(e))


# Create GUI window
root = tk.Tk()
root.title("Previous Exam Score Predictor")

# Create labels and entry fields
hours_studied_label = ttk.Label(root, text="Hours Studied:")
hours_studied_label.grid(row=0, column=0, padx=5, pady=5, sticky="e")
hours_studied_entry = ttk.Entry(root)
hours_studied_entry.grid(row=0, column=1, padx=5, pady=5)

attendance_label = ttk.Label(root, text="Attendance Percentage:")
attendance_label.grid(row=1, column=0, padx=5, pady=5, sticky="e")
attendance_entry = ttk.Entry(root)
attendance_entry.grid(row=1, column=1, padx=5, pady=5)

faculty_feedback_label = ttk.Label(root, text="Faculty Feedback:")
faculty_feedback_label.grid(row=2, column=0, padx=5, pady=5, sticky="e")
faculty_feedback_entry = ttk.Entry(root)
faculty_feedback_entry.grid(row=2, column=1, padx=5, pady=5)

gender_label = ttk.Label(root, text="Gender:")
gender_label.grid(row=3, column=0, padx=5, pady=5, sticky="e")
gender_combobox = ttk.Combobox(root, values=["Male", "Female"])
gender_combobox.grid(row=3, column=1, padx=5, pady=5)

predict_button = ttk.Button(root, text="Predict", command=predict_score)
predict_button.grid(row=4, column=0, columnspan=2, padx=5, pady=5)

root.mainloop()
