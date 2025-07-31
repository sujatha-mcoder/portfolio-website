from django.shortcuts import render,redirect
from mainapp.models import *
from userapp.models import Feedback,Dataset
# from adminapp.models import *
from django.contrib import messages
import time
from django.core.paginator import Paginator
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from django.core.files.storage import default_storage
from django.conf import settings
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input



def userdashboard(req):
    # images_count =  User.objects.all().count()
    # print(images_count)
    # user_id = req.session["User_id"]
    # user = User.objects.get(User_id = user_id)
    return render(req,'user/user-dashboard.html')
  

def profile(req):
    user_id = req.session["User_id"]
    user = User.objects.get(User_id = user_id)
    if req.method == 'POST':
        user_name = req.POST.get('userName')
        user_age = req.POST.get('userAge')
        user_phone = req.POST.get('userPhNum')
        user_email = req.POST.get('userEmail')
        user_address = req.POST.get("userAddress")
        # user_img = request.POST.get("userimg")

        user.Full_name = user_name
        user.Age = user_age
        user.Address = user_address
        user.Phone_Number = user_phone
        user.Email=user_email
       

        if len(req.FILES) != 0:
            image = req.FILES['profilepic']
            user.Image = image
            user.Full_name = user_name
            user.Age = user_age
            user.Address = user_address
            user.Phone_Number = user_phone
            user.Email=user_email
            user.Address=user_address
            
            user.save()
            messages.success(req, 'Updated SUccessfully...!')
        else:
            user.Full_name = user_name
            user.Age = user_age
            user.save()
            messages.success(req, 'Updated SUccessfully...!')
            
    context = {"i":user}
    return render(req, 'user/user-profile.html', context)

def userlogout(req):
    user_id = req.session.get("User_id")

    if user_id:
        try:
            user = User.objects.get(User_id=user_id)
            t = time.localtime()
            current_time = time.strftime('%H:%M:%S', t)
            current_date = time.strftime('%Y-%m-%d', t)

            user.Last_Login_Time = current_time
            user.Last_Login_Date = current_date
            user.save()

            # Clear session
            req.session.flush()
            messages.info(req, 'You are logged out.')
        except User.DoesNotExist:
            messages.warning(req, "User not found.")
    else:
        messages.warning(req, "You are not logged in.")

    return redirect('login')
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from django.shortcuts import render, redirect
from django.contrib import messages
from .models import User, Feedback  # Ensure your models are imported

def userfeedbacks(req):
    id=req.session["User_id"]
    uusser=User.objects.get(User_id=id)
    if req.method == "POST":
        rating=req.POST.get("rating")
        review=req.POST.get("review")
        if not rating:
            messages.info(req,'give rating')
            return redirect('userfeedbacks')
        sid=SentimentIntensityAnalyzer()
        score=sid.polarity_scores(review)
        sentiment=None
        if score['compound']>0 and score['compound']<=0.5:
            sentiment='positive'
        elif score['compound']>=0.5:
            sentiment='very positive'
        elif score['compound']<-0.5:
            sentiment='very negative'
        elif score['compound']<0 and score['compound']>=-0.5:
            sentiment='negative'
        else :
            sentiment='neutral'
        # print(sentiment)        
        # print(rating,feed)
        Feedback.objects.create(Rating=rating, Review=review, Sentiment=sentiment, Reviewer=uusser)
        messages.success(req,'Feedback recorded')
        return redirect('userfeedbacks')
    return render(req,'user/user-feedbacks.html')

from django.shortcuts import render
from django.http import JsonResponse
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

# Load the saved model and scaler
MODEL_FILENAME = "decision_tree_model.pkl"  # Update with your file path
SCALER_FILENAME = "scaler.pkl"  # Update with your file path
dt_model = joblib.load(MODEL_FILENAME)
scaler = joblib.load(SCALER_FILENAME)

# Define label encoder and label mapping
label_order = ['Normal', 'Blackhole', 'Flooding', 'Grayhole', 'TDMA']
label_encoder = LabelEncoder()
label_encoder.fit(label_order)


def predict_attack_type(req):
    if req.method == "POST":
        # Collect input data from the user via the POST request
        input_data = {
            "id": req.POST.get("id"),
            "Time": req.POST.get("Time"),
            "Is_CH": req.POST.get("Is_CH"),
            "who CH": req.POST.get("who CH"),
            "Dist_To_CH": req.POST.get("Dist_To_CH"),
            "ADV_S": req.POST.get("ADV_S"),
            "ADV_R": req.POST.get("ADV_R"),
            "JOIN_S": req.POST.get("JOIN_S"),
            "JOIN_R": req.POST.get("JOIN_R"),
            "SCH_S": req.POST.get("SCH_S"),
            "SCH_R": req.POST.get("SCH_R"),
            "Rank": req.POST.get("Rank"),
            "DATA_S": req.POST.get("DATA_S"),
            "DATA_R": req.POST.get("DATA_R"),
            "Data_Sent_To_BS": req.POST.get("Data_Sent_To_BS"),
            "dist_CH_To_BS": req.POST.get("dist_CH_To_BS"),
            "send_code": req.POST.get("send_code"),
            "Expanded Energy": req.POST.get("Expanded Energy"),
        }

        # Convert input data to a list of floats
        try:
            features = [float(input_data[key]) for key in input_data]
        except ValueError:
            return render(req, "predict_attack_type.html", {"error": "Invalid input. Please enter numeric values for all fields."})

        # Convert to NumPy array and reshape for model input
        features_array = np.array(features).reshape(1, -1)

        # Scale the input data
        scaled_features = scaler.transform(features_array)

        # Predict the attack type
        prediction = dt_model.predict(scaled_features)
        prediction_label = label_encoder.inverse_transform(prediction)[0]

        # Check if the prediction is "Normal" or an attack
        if prediction_label == "Normal":
            attack_status = "Normal (No attack detected)"
        else:
            attack_status = f"Attack detected: {prediction_label}"

        # Predict probabilities for all attack types
        prediction_proba = dt_model.predict_proba(scaled_features)
        proba_dict = {label: round(prob * 100, 2) for label, prob in zip(label_encoder.classes_, prediction_proba[0])}

        # Render the results in the template
        return render(req, "user/user-dos attackdetection.html", {
            "attack_status": attack_status,
            "prediction_probabilities": proba_dict,
            "input_data": input_data,
        })

    return render(req, "user/user-dos attackdetection.html")









        





    
   

