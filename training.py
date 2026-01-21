import cv2
import os
import numpy as np
import joblib

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

IMG_SIZE = 64

DOG_DIR = "DATASETS/DOG"
CAT_DIR = "DATASETS/CAT"

X = []
y = []

for img in os.listdir(CAT_DIR):
    path = os.path.join(CAT_DIR, img)
    image = cv2.imread(path)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image.flatten()
    X.append(image)
    y.append(0)

for img in os.listdir(DOG_DIR):
    path = os.path.join(DOG_DIR, img)
    image = cv2.imread(path)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image.flatten()
    X.append(image)
    y.append(1)

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

svm = SVC(kernel='linear', probability=True)
svm.fit(X_train, y_train)
joblib.dump(svm, "svm_model.pkl")

rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
joblib.dump(rf, "rf_model.pkl")

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
joblib.dump(lr, "lr_model.pkl")

kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X_train)
joblib.dump(kmeans, "kmeans_model.pkl")

print("All models trained successfully!")
