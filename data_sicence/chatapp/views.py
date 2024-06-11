from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from collections import Counter

id = 0
stat = 0
combined_symptoms = {}
callme = {}
implicit_symptom_list = []
implicit_symptom_str = ""

file_path = '/home/minsekan/data_sicence/goal_set.p'
with open(file_path, 'rb') as file:
	consultation_data = pickle.load(file)

test_data = consultation_data['train']

file_path = '/home/minsekan/data_sicence/disease_symptom.p'
with open(file_path, 'rb') as file:
    disease_data = pickle.load(file)

def index(request):
	return render(request, 'chatapp/index.html')

def chat_view(request):
	global stat, implicit_symptom_str, implicit_symptom_list, combined_symptoms, callme, id
	id = 0
	stat = 0
	combined_symptoms = {}
	callme = {}
	implicit_symptom_list = []
	implicit_symptom_str = ""
	return render(request, 'chatapp/chat.html')

# Define NDCG calculation functions
def dcg_score(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gain = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gain / discounts)

def ndcg_score(y_true, y_score, k=10):
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best

# Function to find and print scores based on input
def find_scores(disease_name, symptom_name, df):
	# Find scores based on input
	score_df = df[(df['disease'] == disease_name) & (df['symptom'] == symptom_name)]

	# Sort scores from highest to lowest
	sorted_score_df = score_df.sort_values(by='score', ascending=False)

	# Prepare list to store results
	results = []

	# Append results to the list
	if not sorted_score_df.empty:
		for idx, row in sorted_score_df.iterrows():
			results.append((row['symptom'], row['score']))
	else:
		results.append((symptom_name, 'No scores found'))

	return results

# Function to find and print matching implicit symptoms based on explicit symptoms
def find_matching_implicit_symptoms(explicit_symptoms, df):
	# Matching implicit symptoms
	matching_symptoms = set()

	# Check for partial matching implicit symptoms
	for item in df.itertuples():
		for explicit_symptom in explicit_symptoms:
			explicit_words = explicit_symptom.lower().split()
			implicit_words = item.symptom.lower().split()
			# Check if all explicit words are present in implicit symptom
			if all(word in implicit_words for word in explicit_words) and \
				all(word in explicit_words for word in implicit_words):
				matching_symptoms.add(item.symptom)

	return matching_symptoms

def create_df():
	# Extract explicit and implicit symptoms along with their disease tags
	disease_tags = []
	explicit_symptoms_list = []
	implicit_symptoms_list = []

	for item in test_data:
		disease_tags.append(item['disease_tag'])
		explicit_symptoms = list(item['goal']['explicit_inform_slots'].keys())
		implicit_symptoms = list(item['goal']['implicit_inform_slots'].keys())
		explicit_symptoms_list.append(", ".join(explicit_symptoms))
		implicit_symptoms_list.append(", ".join(implicit_symptoms))

	# Create DataFrame
	df = pd.DataFrame({
		'Disease Tag': disease_tags,
		'Explicit Symptoms': explicit_symptoms_list,
		'Implicit Symptoms': implicit_symptoms_list
	})
	return df

def model_fit(request):
	global implicit_symptom_list
	messages_list = []
	df = create_df()
	# Define and train the Random Forest model
	X = df['Explicit Symptoms'] + " " + df['Implicit Symptoms']
	y = df['Disease Tag']

	label_encoder = LabelEncoder()
	y = label_encoder.fit_transform(y)

	# Split into train and test sets
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	# Vectorize text data
	vectorizer = CountVectorizer()
	X_train = vectorizer.fit_transform(X_train)
	X_test = vectorizer.transform(X_test)

	# Train Random Forest model
	model = RandomForestClassifier(n_estimators=100, random_state=42)
	model.fit(X_train, y_train)
	y_pred = model.predict(X_test)
	accuracy = accuracy_score(y_test, y_pred)
	print(f"Random Forest Accuracy: {accuracy:.2f}")
	messages_list.append({'sender': 'server', 'text': f"Random Forest Accuracy: {accuracy:.2f}"})

	# Calculate NDCG for the model
	y_test_bin = MultiLabelBinarizer().fit_transform([[label] for label in y_test])
	y_score = model.predict_proba(X_test)

	ndcg_scores = [ndcg_score(y_test_bin[i], y_score[i], k=10) for i in range(len(y_test))]
	mean_ndcg = np.mean(ndcg_scores)
	print(f"Random Forest NDCG@10: {mean_ndcg:.5f}")
	messages_list.append({'sender': 'server', 'text': f"Random Forest NDCG@10: {mean_ndcg:.5f}"})
	return JsonResponse({'status': 'Message received', 'message': messages_list})

def input_explicit_symptom(user_input_explicit_symptom):
	global combined_symptoms, callme
	response_message = ""

	# Initialize an empty set to store the unique user input symptoms
	user_input_symptoms = set()

	# Symptom matching and disease prediction logic
	explicit_input = user_input_explicit_symptom.strip().lower()
	input_explicit_symptoms = [symptom.strip() for symptom in explicit_input.split(',')]
	matching_symptoms = {}

	# Find all implicit symptoms associated with the input explicit symptoms
	for item in test_data:
		explicit_info = item.get('goal', {}).get('explicit_inform_slots', {})
		implicit_info = item.get('goal', {}).get('implicit_inform_slots', {})
		disease_tag = item['disease_tag']
		if any(symptom.lower() in [key.lower() for key in explicit_info] for symptom in input_explicit_symptoms):
			matching_symptoms[disease_tag] = matching_symptoms.get(disease_tag, set())
			matching_symptoms[disease_tag].update(implicit_info.keys())

	# Store matching implicit symptoms along with user's explicit symptoms for each disease in the 'callme' memory
	if matching_symptoms:
		for disease_tag, implicit_symptoms in matching_symptoms.items():
			for symptom in implicit_symptoms:
				callme[(disease_tag, symptom)] = input_explicit_symptoms
				# Add the user's input symptoms to the set
				user_input_symptoms.update(input_explicit_symptoms)

	if user_input_symptoms:
		combined_symptoms['User Input Symptoms'] = list(user_input_symptoms)

	# Print the results
	if callme:
		for (disease_tag, symptom), explicit_symptoms in callme.items():
			print(f"Disease: {disease_tag}, Symptom: {symptom}, Explicit Symptoms: {', '.join(explicit_symptoms)}")
	else:
		print("No matching implicit symptoms found.")

	# Print the user input symptoms
	if user_input_symptoms:
		print("User Input Symptoms:", ', '.join(combined_symptoms['User Input Symptoms']))
		response_message += "User Input Symptoms:" + ', '.join(combined_symptoms['User Input Symptoms']) + "\n"
	else:
		print("No user input symptoms provided.")
		response_message += "No user input symptoms provided."
	
	# Convert the data into a DataFrame
	data = {
		'disease': [],
		'symptom': [],
		'score': []
	}

	for disease, info in disease_data.items():
		for symptom, score in info['symptom'].items():
			data['disease'].append(disease)
			data['symptom'].append(symptom)
			data['score'].append(score)

	df = pd.DataFrame(data)

	# Process each input from the callme memory and store results
	all_results = []
	for (disease_name, symptom_name), explicit_symptoms in callme.items():
		all_results.extend(find_scores(disease_name, symptom_name, df))

	# Sort all results by score in descending order, while handling 'No scores found'
	all_results.sort(key=lambda x: float(x[1]) if isinstance(x[1], float) else 0, reverse=True)

	# Filter out results with 'No scores found'
	filtered_results = [result for result in all_results if result[1] != 'No scores found']

	# Print only the first 10 results with order number
	for idx, (symptom, score) in enumerate(filtered_results[:10], start=1):
		implicit_symptom_list.append(symptom)
		print(f"{idx}. {symptom}, {score}")
	return response_message



def input_implicit_symptom(user_input_implicit_symptom):
	response_message = ""

	# Convert the data into a DataFrame
	data = {
		'disease': [],
		'symptom': [],
		'score': []
	}

	for disease, info in disease_data.items():
		for symptom, score in info['symptom'].items():
			data['disease'].append(disease)
			data['symptom'].append(symptom)
			data['score'].append(score)

	df = pd.DataFrame(data)

	# Initialize a set to store the matched implicit symptoms
	matching_implicit_symptoms = set()

	# Take user input for implicit symptoms
	implicit_input = user_input_implicit_symptom

	# Split the input into individual implicit symptoms
	implicit_symptoms = [symptom.strip() for symptom in implicit_input.split(',')]

	# Find matching implicit symptoms based on user input
	matching_implicit_symptoms = find_matching_implicit_symptoms(implicit_symptoms, df)

	# Check if any matching implicit symptoms are found
	if matching_implicit_symptoms:
		response_message += "\n입력받은 암시적 증상 : \n"
		print("\nMatched Implicit Symptoms:")
		for symptom in matching_implicit_symptoms:
			response_message += f"- {symptom}\n"
			print(f"- {symptom}")

		# Update the combined_symptoms dictionary to include the matched implicit symptoms
		if 'Matched Implicit Symptoms' in combined_symptoms:
			combined_symptoms['Matched Implicit Symptoms'].update(matching_implicit_symptoms)
		else:
			combined_symptoms['Matched Implicit Symptoms'] = matching_implicit_symptoms
	else:
		print("No matching implicit symptoms found.")
		response_message += "No matching implicit symptoms found.\n"
	print(combined_symptoms)
	return response_message

def get_disease():
	response_message = ""
	df = create_df()
	# Define and train the Random Forest model
	X = df['Explicit Symptoms'] + " " + df['Implicit Symptoms']
	y = df['Disease Tag']

	label_encoder = LabelEncoder()
	y = label_encoder.fit_transform(y)

	# Vectorize text data
	vectorizer = CountVectorizer()
	X_vectorized = vectorizer.fit_transform(X)

	# Train Random Forest model
	model = RandomForestClassifier(n_estimators=100, random_state=42)
	model.fit(X_vectorized, y)

	# Function to predict diseases based on combined symptoms
	def predict_diseases(combined_symptoms, top_n=5):
		symptoms_vectorized = vectorizer.transform([combined_symptoms])
		probas = model.predict_proba(symptoms_vectorized)[0]
		top_indices = probas.argsort()[-top_n:][::-1]
		predicted_diseases = label_encoder.inverse_transform(top_indices)
		return predicted_diseases

	# Assuming combined_symptoms is a dictionary containing symptoms
	# For example: combined_symptoms = {'symptoms': ['knee pain', 'fever', 'headache']}
	# combined_symptoms_dict = {'symptoms': ['knee pain']}  # Example
	combined_symptoms_dict = combined_symptoms

	# Convert the set of combined symptoms to a single string
	combined_symptoms_str = ", ".join(combined_symptoms_dict['User Input Symptoms']).lower()
	combined_symptoms_str += ", "
	combined_symptoms_str += ", ".join(combined_symptoms_dict['Matched Implicit Symptoms']).lower()
	print(combined_symptoms_str)

	# Predict top 5 diseases based on combined symptoms
	top_5_diseases = predict_diseases(combined_symptoms_str)

	# Count occurrences of each disease prediction
	disease_counts = Counter(top_5_diseases)

	# Sort diseases by count in descending order
	sorted_diseases = sorted(disease_counts.items(), key=lambda x: x[1], reverse=True)

	# Print the top 5 predicted diseases
	response_message += "\nTop 5 Predicted Diseases : \n"
	print("Top 5 Predicted Diseases:")
	for idx, (disease, count) in enumerate(sorted_diseases[:5], start=1):
		response_message += f"{idx}. {disease} (Count: {count})\n"
		print(f"{idx}. {disease} (Count: {count})")
	return response_message

@csrf_exempt
def send_message(request):
	global stat, id, implicit_symptom_str, implicit_symptom_list
	if request.method == 'POST':
		data = json.loads(request.body)
		message = data.get('message')
		if stat == 0:
			response_message = input_explicit_symptom(message)
			stat = 1
		if stat == 2:
			if message.lower() == "yes":
				if implicit_symptom_str:
					implicit_symptom_str += ", " + implicit_symptom_list[id]
				else:
					implicit_symptom_str += implicit_symptom_list[id]
			id += 1
			stat = 1
		if stat == 1:
			print(id, len(implicit_symptom_list), implicit_symptom_str)
			if id < len(implicit_symptom_list):
				response_message = implicit_symptom_list[id] + " 증상이 있으십니까? (yes or no)\n"
				stat = 2
			else:
				stat = 3
		if stat == 3:
			response_message = input_implicit_symptom(implicit_symptom_str)
			response_message += get_disease()
		
		return JsonResponse({'status': 'Message received', 'message': response_message})
	else:
		return JsonResponse({'status': 'Invalid request'}, status=400)
