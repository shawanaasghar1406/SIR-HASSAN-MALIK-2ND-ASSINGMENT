# SIR-HASSAN-MALIK-2ND-ASSINGMENT
from sklearn.feature_extraction.text import CountVectorizer

document = ["How Are You Ms XD SHAWANA ASGHAR ",
			"It's Pleasure To Meet You",
			"So You Work In A Software Company"]

# Create a Vectorizer Object
vectorizer = CountVectorizer()

vectorizer.fit(document)

# Printing the identified Unique words along with their indices
print("Vocabulary: ", vectorizer.vocabulary_)

# Encode the Document
vector = vectorizer.transform(document)

# Summarizing the Encoded Texts
print("Encoded Document is:")
print(vector.toarray())
