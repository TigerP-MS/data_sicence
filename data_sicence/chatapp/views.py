from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json

# 전역 변수로 메시지 리스트와 입력 횟수를 초기화합니다.
messages_list = []
list = []
input_count = 0

def index(request):
    return render(request, 'chatapp/index.html')

def chat_view(request):
    global messages_list, list, input_count
    # 페이지를 새로 열 때마다 리스트와 입력 횟수를 초기화합니다.
    messages_list = []
    list = []
    input_count = 0
    return render(request, 'chatapp/chat.html')

@csrf_exempt 
def add_to_chat(request):
    global messages_list, list, input_count
    if request.method == 'POST':
        data = json.loads(request.body)
        user_message = data.get('message', '')

        if user_message and input_count < 10:
            input_count += 1
            messages_list.append({'sender': 'user', 'text': user_message})
            list.append(user_message)
            server_response = f"현재 데이터: {', '.join(list)}\n입력 횟수: {input_count}"
            messages_list.append({'sender': 'server', 'text': server_response})

            # 입력 횟수가 10번일 때 특별한 메시지를 추가합니다.
            if input_count == 10:
                special_message = f'입력받은 데이터는 {", ".join(list)}입니다.'
                messages_list.append({'sender': 'server', 'text': special_message})

        return JsonResponse({'messages': messages_list, 'input_count': input_count})

    return JsonResponse({'messages': messages_list, 'input_count': input_count})
