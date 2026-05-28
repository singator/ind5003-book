def word_analogy(model, nn_obj, word_a, word_b, word_c):
    """
    Solve the analogy: word_a is to word_b as word_c is to ____
    
    Using sklearn's NearestNeighbors (KNN) to find the closest embedding.
    
    Example: word_a="man", word_b="woman", word_c="king"
    This solves: "man is to woman as king is to ____"
    
    Algorithm:
    1. Get embeddings for word_a, word_b, word_c
    2. Compute relationship: word_b - word_a
    3. Apply to word_c: word_c + relationship = target_vector
    4. Use KNN to find nearest neighbors to target_vector
    5. Return top matches (excluding query words)
    
    Parameters:
    -----------
    model : StaticVectors
        The GloVe embedding model
    word_a : str
        First word in the analogy (e.g., "man")
    word_b : str
        Second word in the analogy (e.g., "woman")
    word_c : str
        Third word in the analogy (e.g., "king")
    topn : int
        Number of closest matches to return
    
    Returns:
    --------
    results : list of tuples
        List of (word, distance_score) pairs, sorted by proximity
    """
    
    # Step 1: Get embeddings for the three words
    try:
        emb_a = model.embeddings([word_a])[0]  # man
        emb_b = model.embeddings([word_b])[0]  # woman
        emb_c = model.embeddings([word_c])[0]  # king
    except Exception as e:
        print(f"Error: One of the words not found in vocabulary: {e}")
        return []

    #org_indices
    
    # Step 2: Compute the relationship vector (word_b - word_a)
    # This captures what it means to go from "man" to "woman"
    relationship = emb_b - emb_a
    
    # Step 3: Add the relationship to word_c
    # king + (woman - man) = target vector
    target_vector = emb_c + relationship
    
    # Normalize the target vector
    # target_vector = target_vector / np.linalg.norm(target_vector)
    
    # Step 5: Query KNN to find nearest neighbors
    distances, indices = nn_obj.kneighbors(target_vector.reshape(1, -1), n_neighbors=5)

    all_tokens = list(model.tokens.items())
    if indices is not None:
        out_tokens = [all_tokens[ii][0] for ii in indices[0]]
        #out_tokens.append(all_tokens[indices[0]
    #    return list(model.tokens.items())[indices.item()][0]
    to_return  = [xxx for xxx in out_tokens if xxx not in [word_a, word_b, word_c]]
    
    return to_return
