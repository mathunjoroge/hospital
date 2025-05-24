import spacy
nlp = spacy.load("en_core_sci_sm")
nlp.add_pipe("scispacy_linker", config={"linker_name": "umls"})
doc = nlp("Test text")
print(doc.ents)