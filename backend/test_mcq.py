import requests
import json

print("Testing MCQ generation...")
r = requests.post('http://localhost:8000/student/mcq', json={'subject': 'FDS', 'unit': 'unit 1', 'count': 1}, timeout=120)
print(f'Status: {r.status_code}')
data = r.json()
print(f'MCQ status: {data.get("status")}')
print(f'MCQ count: {data.get("count", 0)}')
if data.get("mcqs"):
    for i, mcq in enumerate(data["mcqs"]):
        print(f'\nMCQ {i+1}:')
        print(f'  Question: {mcq["question"][:100]}')
        print(f'  Options: {mcq["options"]}')
        print(f'  Answer: {mcq["correct_answer"]}')
        print(f'  Explanation: {mcq.get("explanation", "")[:100]}')
else:
    print(f'No MCQs parsed!')
    print(f'Message: {data.get("message", "N/A")}')
    print(f'Full response: {json.dumps(data, indent=2)[:500]}')
