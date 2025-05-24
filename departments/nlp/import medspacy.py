from nlp_pipeline import get_nlp
nlp = get_nlp()
print(nlp.pipe_names)  # Should include medspacy_context and scispacy_linker without errors