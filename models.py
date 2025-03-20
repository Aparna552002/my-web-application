from django.db import models

class AutismPrediction(models.Model):
    GENDER_CHOICES = [
        ("Male", "Male"),
        ("Female", "Female"),
        ("Other", "Other"),
    ]

    PREDICTION_CHOICES = [
        ("Positive Autism", "Positive Autism"),
        ("Negative Autism", "Negative Autism"),
    ]

    name = models.CharField(max_length=100)
    age = models.PositiveIntegerField()  # Ensures age is non-negative
    gender = models.CharField(max_length=10, choices=GENDER_CHOICES)
    ethnicity = models.CharField(max_length=50)
    scores = models.JSONField()  # Stores list directly as JSON
    prediction = models.CharField(max_length=50, choices=PREDICTION_CHOICES)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.name} ({self.age}, {self.gender}) - {self.prediction} [{self.created_at}]"
