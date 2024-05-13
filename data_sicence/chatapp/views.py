from django.shortcuts import render, redirect
from .models import Message
from django.utils import timezone

def index(request):
    messages = Message.objects.all().order_by('-created_at')[:10]
    return render(request, 'chatapp/index.html', {'messages': messages})

def send_message(request):
    text = request.POST.get('text', '')
    if text:
        Message.objects.create(text=text, created_at=timezone.now())
    return redirect('index')
