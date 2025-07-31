from django.db import models

# Create your models here.
class manage_users_model(models.Model):
    User_id = models.AutoField(primary_key = True)
    user_Profile = models.FileField(upload_to = 'images/')
    User_Email = models.EmailField(max_length = 50)
    User_Status = models.CharField(max_length = 10)
    
    class Meta:
        db_table = 'manage_users'

from django.db import models

class SVM(models.Model):
    precision = models.FloatField()
    recall = models.FloatField()
    f1_score = models.FloatField()
    accuracy = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)  # To track when the metrics were saved

    def __str__(self):
        return f"Metrics at {self.created_at}: Accuracy {self.accuracy:.2f}"
    
    class Meta:
        db_table = 'SVM'


from django.db import models

class DT(models.Model):
    precision = models.FloatField()
    recall = models.FloatField()
    f1_score = models.FloatField()
    accuracy = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)


    def __str__(self):
        return f"Decision Tree Results - {self.name}"

    class Meta:
        db_table = 'DT'  # Custom database table name



