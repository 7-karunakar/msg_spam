import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
from sklearn.svm import SVC

data = pd.read_csv('spam.csv', encoding='latin1')


X = data['v2']
y = data['v1'].apply(lambda x: 1 if x == 'spam' else 0)  # Convert labels to binary (0 for ham, 1 for spam)


vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#model = LogisticRegression()
model = SVC(kernel = 'linear')

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

print(classification_report(y_test, y_pred))

joblib.dump(model, 'spam_modelsvm.pkl')
joblib.dump(vectorizer, 'vectorizersvm.pkl')
