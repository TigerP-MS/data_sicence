from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
	path('chat/', views.chat_view, name='chat'),
    path('add_to_chat/', views.add_to_chat, name='add_to_chat'),
]
