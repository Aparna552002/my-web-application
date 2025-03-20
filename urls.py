from django.urls import path
from . import views

urlpatterns = [
    path("", views.custom_login, name="login"),  # Set login as the default page
    path("home/", views.home, name="home"),  # Ensure home has a separate path
    path("register/", views.register, name="register"),  # Added register path
    path("predict/", views.predict, name="predict"),
]
