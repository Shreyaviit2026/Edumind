import requests
import json

print("Testing ask question API...")
r = requests.post('http://localhost:8000/student/ask', json={'subject': 'Cloud', 'unit': 'Unit 1', 'question': 'what is data center?'}, timeout=120)
print(f'Status: {r.status_code}')
data = r.json()
print(f'Ask status: {data.get("status")}')
print(f'Message: {data.get("message", "N/A")}')
print(f'Answer: {data.get("answer", "N/A")}')
