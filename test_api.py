import requests

url = "https://test1-mqan.onrender.com/hackrx/run"
headers = {
    "Authorization": "Bearer d464d88731074c5923019b6139916b4ba2e7cd1b8cb01316fed78295b75c066e",
    "Content-Type": "application/json",
    "Accept": "application/json"
}

data = {
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
    "questions": [
        "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
        "What is the waiting period for pre-existing diseases (PED) to be covered?",
        "Does this policy cover maternity expenses, and what are the conditions?",
        "What is the waiting period for cataract surgery?",
        "Are the medical expenses for an organ donor covered under this policy?",
        "What is the No Claim Discount (NCD) offered in this policy?",
        "Is there a benefit for preventive health check-ups?",
        "How does the policy define a 'Hospital'?",
        "What is the extent of coverage for AYUSH treatments?",
        "Are there any sub-limits on room rent and ICU charges for Plan A?"
    ]
}

response = requests.post(url, headers=headers, json=data)
print("Status:", response.status_code)

if response.status_code == 200:
    res = response.json()
    print("\nAnswers:")
    for idx, ans in enumerate(res["answers"], 1):
        print(f"{idx}. {ans}")
    print(f"\nTotal Response Time: {res['total_response_time_sec']} seconds")
else:
    print("Error Response:", response.text)
