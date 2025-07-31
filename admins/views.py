from django.shortcuts import render,redirect
from mainapp.models import*
from userapp.models import*
from admins.models import *
from django.contrib import messages
from django.core.paginator import Paginator
from django.http import HttpResponse
import os
import shutil
from django.shortcuts import render
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from django.contrib import messages
from .models import SVM,DT
from keras.models import load_model


#gradient boost machine algo for getting acc ,precession , recall , f1 score
# Create your views here.
def adminlogout(req):
    # messages.info(req,'You are logged out...!')
    return redirect('admin')

def admindashboard(req):
    all_users_count =  User.objects.all().count()
    pending_users_count = User.objects.filter(User_Status = 'Pending').count()
    rejected_users_count = User.objects.filter(User_Status = 'removed').count()
    accepted_users_count =User.objects.filter(User_Status = 'accepted').count()
    Feedbacks_users_count= Feedback.objects.all().count()
    user_uploaded_images =Dataset.objects.all().count()
    return render(req,'admin/admin-dashboard.html',{'a' : all_users_count, 'b' : pending_users_count, 'c' : rejected_users_count, 'd' : accepted_users_count, 'e':Feedbacks_users_count, 'f':user_uploaded_images})

def pendingusers(req):
    pending = User.objects.filter(User_Status = 'Pending')
    paginator = Paginator(pending, 5) 
    page_number = req.GET.get('page')
    post = paginator.get_page(page_number)
    return render(req,'admin/admin-pending-users.html', { 'user' : post})

def delete_user(req, id):
    User.objects.get(User_id = id).delete()
    messages.warning(req, 'User was Deleted..!')
    return redirect('manageusers')

def accept_user(req, id):
    status_update = User.objects.get(User_id = id)
    status_update.User_Status = 'accepted'
    status_update.save()
    messages.success(req, 'User was accepted..!')
    return redirect('pendingusers')

def manageusers(req):
    manage_users  = User.objects.all()
    paginator = Paginator(manage_users, 5)
    page_number = req.GET.get('page')
    post = paginator.get_page(page_number)
    return render(req, 'admin/admin-manage-users.html', {"allu" : manage_users, 'user' : post})

def reject_user(req, id):
    status_update2 = User.objects.get(User_id = id)
    status_update2.User_Status = 'removed'
    status_update2.save()
    messages.warning(req, 'User was Rejected..!')
    return redirect('pendingusers')

def admin_datasetupload(req):
    return render(req,'admin/admin-upload-dataset.html')

def admin_dataset_btn(req):
    messages.success(req, 'Dataset uploaded successfully..!')
    return redirect('admin_datasetupload')



def adminfeedback(req):
    feed =Feedback.objects.all()
    return render(req,'admin/user-feedback.html', {'back':feed})

def adminsentiment(req):
    fee = Feedback.objects.all()
    return render(req,'admin/user-sentiment.html' , {'cat':fee})

def usergraph(req):
    positive = Feedback.objects.filter(Sentiment = 'positive').count()
    very_positive = Feedback.objects.filter(Sentiment = 'very positive').count()
    negative = Feedback.objects.filter(Sentiment = 'negative').count()
    very_negative = Feedback.objects.filter(Sentiment = 'very negative').count()
    neutral = Feedback.objects.filter(Sentiment = 'neutral').count()
    context ={
        'vp': very_positive, 'p':positive, 'n':negative, 'vn':very_negative, 'ne':neutral
    }
    return render(req,'admin/user-sentiment-graph.html',context)

def SVM_alg(req):
  return render(req,'admin/SVM_alg.html')

def DT_alg(req):
  return render(req,'admin/DT_alg.html')

from django.shortcuts import render
from .models import SVM  # Import the model
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

def SVM_btn(req):
    # Load dataset
    file_path = "dataset/WSN-DS.csv"  # Replace with your actual file path
    df = pd.read_csv(file_path)

    # Display first few rows for inspection (optional for debugging)
    print(df.head())

    # Encode categorical target variable
    label_encoder = LabelEncoder()
    df['Attack type'] = label_encoder.fit_transform(df['Attack type'])

    # Separate features and target
    X = df.drop(['Attack type'], axis=1)
    y = df['Attack type']

    # Split dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Standardize the feature values
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train SVM model
    svm_model = SVC(kernel='linear', random_state=42)
    svm_model.fit(X_train, y_train)

    # Predict on test set
    y_pred = svm_model.predict(X_test)

    # Calculate metrics
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    accuracy = accuracy_score(y_test, y_pred)

    # Save metrics in the database
    svm_metric = SVM.objects.create(
        precision=precision,
        recall=recall,
        f1_score=f1,
        accuracy=accuracy
    )

    data1=SVM.objects.last()

    # Print saved metrics
    print(f"Metrics saved to database: Precision={svm_metric.precision:.2f}, "
          f"Recall={svm_metric.recall:.2f}, F1-Score={svm_metric.f1_score:.2f}, "
          f"Accuracy={svm_metric.accuracy:.2f}")

    # Prepare metrics data to send to the template
    data = {
        'precision': f"{precision:.2f}",
        'recall': f"{recall:.2f}",
        'f1_score': f"{f1:.2f}",
        'accuracy': f"{accuracy:.2f}"
    }

    # Save the trained model
    model_filename = "svm_model.pkl"
    joblib.dump(svm_model, model_filename)
    print(f"Model saved to {model_filename}")

    # Render the results in the template
    return render(req, 'admin/SVM_alg.html', data)

from django.shortcuts import render
from .models import DT  # Import the model
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

def DT_btn(req):
    # Load dataset
    file_path = "dataset/WSN-DS.csv"  # Replace with your actual file path
    df = pd.read_csv(file_path)

    # Display first few rows for inspection (optional for debugging)
    print(df.head())

    # Encode categorical target variable
    label_encoder = LabelEncoder()
    df['Attack type'] = label_encoder.fit_transform(df['Attack type'])

    # Separate features and target
    X = df.drop(['Attack type'], axis=1)
    y = df['Attack type']

    # Split dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Standardize the feature values
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train Decision Tree model
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train, y_train)

    # Predict on test set
    y_pred = dt_model.predict(X_test)

    # Calculate metrics
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    accuracy = accuracy_score(y_test, y_pred)

    # Save metrics in the database
    dt_metric = DT.objects.create(
        precision=precision,
        recall=recall,
        f1_score=f1,
        accuracy=accuracy
    )

    # Retrieve the last saved metrics
    data1 = DT.objects.last()

    # Print saved metrics
    print(f"Metrics saved to database: Precision={dt_metric.precision:.2f}, "
          f"Recall={dt_metric.recall:.2f}, F1-Score={dt_metric.f1_score:.2f}, "
          f"Accuracy={dt_metric.accuracy:.2f}")

    # Prepare metrics data to send to the template
    data = {
        'precision': f"{precision:.2f}",
        'recall': f"{recall:.2f}",
        'f1_score': f"{f1:.2f}",
        'accuracy': f"{accuracy:.2f}"
    }

    # Save the trained model
    model_filename = "decision_tree_model.pkl"
    joblib.dump(dt_model, model_filename)
    print(f"Model saved to {model_filename}")

    # Save scaler for future use
    scaler_filename = "scaler.pkl"
    joblib.dump(scaler, scaler_filename)
    print(f"Scaler saved to {scaler_filename}")

    # Render the results in the template
    return render(req, 'admin/DT_alg.html', data)





def admingraph(req):
    details1 = SVM.objects.last()
    if details1:
       SVM1 = details1.accuracy
    else:
        SVM1 = 0



    details3 = DT.objects.last()
    if details3:
       DT1 = details3.accuracy
    else:
        DT1 = 0

    print('SVM1:',SVM1)
    print('DT1:',DT1)
    return render(req,'admin/admin-graph-analysis.html',{'SVM':SVM1, 'DT':DT1})

    




















