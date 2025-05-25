from pymongo import MongoClient

client = MongoClient('mongodb://localhost:27017')
db = client['clinical_db']
collection = db['symptoms']

# Drop the existing index
collection.drop_index('category_1_symptom_1')
print("Dropped index 'category_1_symptom_1'")

# Recreate with unique=True
collection.create_index([('category', 1), ('symptom', 1)], unique=True, name='category_1_symptom_1')
print("Created unique index 'category_1_symptom_1'")

client.close()