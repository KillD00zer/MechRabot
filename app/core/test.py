import requests

# ضع الـ API Key الجديد الخاص بك هنا بين علامات التنصيص
api_key = "sk-460rH9gP9Y78EZUlM4ZeEon328Y8oidHwjCdv9An6hTd2goB"

url = "https://api.deepseek.com/chat/completions"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

data = {
    # استخدمنا الاسم القياسي للموديل
    "model": "deepseek-chat", 
    # غيرنا الرسالة للغة الإنجليزية
    "messages": [{"role": "user", "content": "Explain the concept of an engine timing belt in one short sentence."}]
}

print("جاري الاتصال بـ AgentRouter...")
response = requests.post(url, headers=headers, json=data)

print("\n=== النتيجة ===")
print(f"Status Code: {response.status_code}")
print(f"Response Text: {response.text}")
