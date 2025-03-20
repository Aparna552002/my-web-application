from django.http import JsonResponse
from django.shortcuts import render, redirect
import json
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth import authenticate, login
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from .models import AutismPrediction  # Import your model


def custom_login(request):
    """Handles user login."""
    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")

        # Debugging: Print input values
        print(f"Attempting login: {username}, {password}")

        user = authenticate(request, username=username, password=password)

        if user is not None:
            login(request, user)
            messages.success(request, "Login successful!")
            return redirect("home")  # Redirect to home after login
        else:
            messages.error(request, "Invalid username or password")
    
    return render(request, "login.html")  # Show login page


def register(request):
    """Handles user registration"""
    if request.method == "POST":
        username = request.POST.get("username")
        email = request.POST.get("email")
        password1 = request.POST.get("password1")
        password2 = request.POST.get("password2")

        # Ensure all fields are filled
        if not username or not email or not password1 or not password2:
            messages.error(request, "All fields are required.")
            return redirect("register")

        # Check if passwords match
        if password1 != password2:
            messages.error(request, "Passwords do not match.")
            return redirect("register")

        # Check if the username already exists
        if User.objects.filter(username=username).exists():
            messages.error(request, "Username already taken. Try another.")
            return redirect("register")

        # Create and save the user
        user = User.objects.create_user(username=username, email=email, password=password1)
        user.save()

        messages.success(request, "Registration successful! You can now log in.")
        return redirect("login")  # Redirect to login page after registration

    return render(request, "register.html")


@login_required(login_url="login")  # Redirects unauthenticated users
def home(request):
    return render(request, "home.html")


@csrf_exempt
@login_required  # Protects the prediction view (optional)
def predict(request):
    """Handles autism prediction based on user input."""
    if request.method == "POST":
        try:
            data = json.loads(request.body.decode("utf-8"))

            # Extract and validate required fields
            name = data.get("name", "Anonymous")
            age = data.get("age")
            gender = data.get("gender", "Unknown")
            ethnicity = data.get("ethnicity", "Unknown")
            scores = [data.get(f"a{i}", -1) for i in range(1, 11)]

            # Ensure all required fields are present
            if age is None or not isinstance(age, int) or age <= 0:
                return JsonResponse({"error": "Invalid age provided."}, status=400)

            if any(score not in [0, 1] for score in scores):
                return JsonResponse({"error": "Scores must be 0 or 1 only."}, status=400)

            # **Prediction Logic**
            count_ones = scores.count(1)
            count_zeros = scores.count(0)

            if count_ones > count_zeros:
                prediction = "Positive Autism"
            else:
                prediction = "Negative Autism"

            # **Save Prediction to Database**
            autism_prediction = AutismPrediction.objects.create(
                name=name,
                age=age,
                gender=gender,
                ethnicity=ethnicity,
                scores=scores,
                prediction=prediction,
            )

            print("Saved:", autism_prediction)  # Debugging log

            return JsonResponse({"prediction": prediction})

        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON format."}, status=400)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=400)

    return JsonResponse({"error": "Invalid request method"}, status=405)
