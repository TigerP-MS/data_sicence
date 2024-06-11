from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
	path('chat/', views.chat_view, name='chat'),
	path('model_fit/', views.model_fit, name='model_fit'),
	path('send_message/', views.send_message, name='send_message')
]
