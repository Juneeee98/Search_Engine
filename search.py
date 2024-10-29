import lucene
import time
import math
import numpy as np
from math import log10, log
from collections import Counter
from spacy.lang.en import stop_words
from org.apache.lucene.analysis.tokenattributes import CharTermAttribute
from org.apache.lucene.store import FSDirectory
from org.apache.lucene.index import MultiReader, DirectoryReader, Term, IndexWriter, IndexWriterConfig
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.analysis.en import EnglishAnalyzer
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.search import TermQuery, IndexSearcher, BooleanQuery, BooleanClause
from org.apache.lucene.document import DoublePoint, StringField, Document
from org.apache.lucene.analysis import CharArraySet
from java.util import Arrays
from java.nio.file import Paths
import shutil
import os
import re
import matplotlib.pyplot as plt


def calculate_distance(lat1, lon1, lat2, lon2):
    """
    Calculates the Haversine distance between two latitude/longitude points.
    """
    R = 6371  # Earth radius in kilometers
    lat1_rad, lon1_rad = math.radians(lat1), math.radians(lon1)
    lat2_rad, lon2_rad = math.radians(lat2), math.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = R * c
    return distance

def calculate_log_weight(value):
    """
    Calculates a log-based weight for a given value.
    """
    return 1 + log10(value) if value > 0 else 0

def search_by_business_name(searcher, query_string, N):
    """
    Searches for businesses by their name and provides star distribution for each result.
    Custom scoring is applied based on star ratings and review count.
    """
    start_time = time.time()
    analyzer = StandardAnalyzer()

    # Execute keyword search for business name
    query = QueryParser("name", analyzer).parse(query_string)

    # Calculate max_results based on term frequencies
    max_freq = 0
    terms = query_string.split()
    for term in terms:
        # Get document frequency of each term
        term_query = Term("name", term)
        max_freq = max(max_freq, searcher.getIndexReader().docFreq(term_query))

    # Set max_results based on frequency; higher frequency terms lead to higher max_results
    max_results = max(N, min(1000, N * max(1, max_freq // 10)))

    hits = searcher.search(query, max_results)

    results = []
    seen_business_ids = set()  # To track unique business IDs

    for hit in hits.scoreDocs:
        doc = searcher.doc(hit.doc)
        business_id = doc.get("business_id")
        address = doc.get("address") or "N/A"
        city = doc.get("city") or "N/A"
        state = doc.get("state") or "N/A"
        postal_code = doc.get("postal_code") or "N/A"
        
        # Skip if this business_id has already been seen
        if business_id in seen_business_ids:
            continue

        name = doc.get("name") or "N/A"
        stars = float(doc.get("stars") or "0")  # Convert stars to float
        review_count = int(doc.get("review_count") or "0")  # Convert review count to int
        score = hit.score

        # Apply log-based weights for stars and review_count
        log_star_weight = calculate_log_weight(stars)
        log_review_count_weight = calculate_log_weight(review_count)

        # Adjust score based on logarithmic scaling of stars and review count
        adjusted_score = score * (1 + (log_star_weight + log_review_count_weight) / 10)

        result = {
            "business_id": business_id,
            "name": name,
            "address": address,
            "city": city,
            "state": state,
            "postal_code": postal_code,
            "stars": stars,
            "review_count": review_count,
            "lucene_score": score,
            "adjusted_score": adjusted_score  # Use custom adjusted score
        }
        results.append(result)

        # Mark this business_id as seen to avoid duplicates
        seen_business_ids.add(business_id)

    # Sort results by the adjusted score in descending order
    results.sort(key=lambda x: x['adjusted_score'], reverse=True)

    # Return only the top N results after filtering
    end_time = time.time()
    print(f"\nQuery took {end_time - start_time:.4f} seconds")

    return results[:N]

def search_by_review_text_with_business(searcher, query_string, N):
    """
    Searches for reviews by keywords in the review text and retrieves associated business names.
    Custom scoring is applied based on keyword relevance, usefulness, coolness, funniness, and business ratings.
    """
    start_time = time.time()
    analyzer = StandardAnalyzer(EnglishAnalyzer.ENGLISH_STOP_WORDS_SET)

    # Retrieve all matching documents (using a large value for maxResults)
    query = QueryParser("review_text", analyzer).parse(query_string)

    # Tokenize query string to filter out stop words
    token_stream = analyzer.tokenStream("field", query_string)
    token_stream.reset()

    # Extract non-stop words from the token stream
    terms = []
    while token_stream.incrementToken():
        term_text = token_stream.getAttribute(CharTermAttribute.class_).toString()
        terms.append(term_text)
    
    # Close the token stream
    token_stream.end()
    token_stream.close()

    # Determine max_results based on document frequency of terms in the query
    max_freq = 0
    for term in terms:
        term_query = Term("review_text", term)
        max_freq = max(max_freq, searcher.getIndexReader().docFreq(term_query))

    # Set max_results based on term frequency
    max_results = max(N, min(1000, N * max(1, max_freq // 10)))

    hits = searcher.search(query, max_results)

    # Separate results with and without valid business names
    results_with_business_name = []
    results_without_business_name = []
    
    # Keep track of unique reviews by their review_id
    seen_review_ids = set()

    # Define weights for useful, cool, and funny votes
    useful_weight = 1.0
    cool_weight = 0.5
    funny_weight = 0.2

    for hit in hits.scoreDocs:
        doc = searcher.doc(hit.doc)
        review_id = doc.get("review_id")  # Assuming review_id is unique for each review
        business_id = doc.get("business_id")
        user_id = doc.get("user_id")
        
        # Skip duplicates based on review_id
        if review_id in seen_review_ids:
            continue
        
        review_text = doc.get("review_text") or "N/A"
        useful = int(doc.get("useful") or "0")  # Convert useful to int
        cool = int(doc.get("cool") or "0")  # Convert cool to int
        funny = int(doc.get("funny") or "0")  # Convert funny to int
        score = hit.score  # Lucene's keyword relevance score

        # Use TermQuery to exactly match the business_id field
        business_query = TermQuery(Term("business_id", business_id))
        business_hits = searcher.search(business_query, 1)

        business_name = "N/A"
        stars = 0.0  # Default stars for businesses without a rating

        # Fetch business name and rating if the business exists in the index
        if business_hits.totalHits.value > 0:
            business_doc = searcher.doc(business_hits.scoreDocs[0].doc)
            business_name = business_doc.get("name") or "N/A"
            stars = float(business_doc.get("stars") or "0")

        # Apply log-based weights
        log_useful_weight = calculate_log_weight(useful) * useful_weight
        log_cool_weight = calculate_log_weight(cool) * cool_weight
        log_funny_weight = calculate_log_weight(funny) * funny_weight
        log_star_weight = calculate_log_weight(stars)

        # Custom scoring with log-transformed weights
        adjusted_score = score * (1 + (log_useful_weight + log_cool_weight + log_funny_weight + log_star_weight) / 10 )

        result = {
            "business_id": business_id,
            "review_id": review_id,  # Adding review_id to track unique results
            "user_id": user_id,
            "name": business_name,
            "review_text": review_text,
            "useful": useful,
            "cool": cool,
            "funny": funny,
            "stars": stars,
            "lucene_score": score,
            "adjusted_score": adjusted_score  # Use custom adjusted score
        }

        # Separate reviews with valid business names
        if business_name != "N/A":
            results_with_business_name.append(result)
        else:
            results_without_business_name.append(result)

        # Add the review_id to the seen set to prevent duplicates
        seen_review_ids.add(review_id)

    # Combine results with business name first
    prioritized_results = results_with_business_name + results_without_business_name

    # Sort by business name presence (1 if name exists, 0 otherwise) and then by the score
    prioritized_results.sort(key=lambda x: (1 if x["name"] != "N/A" else 0, x["adjusted_score"]), reverse=True)
    end_time = time.time()

    print(f"\nQuery took {end_time - start_time:.4f} seconds")

    # Return only the top N results
    return prioritized_results[:N]

def geospatial_search(searcher, lat_min, lat_max, lon_min, lon_max, N):
    """
    Searches for businesses within a given geospatial bounding box and provides star distribution and address information.
    Custom scoring is applied based on proximity to the center, stars, and review count.
    """
    start_time = time.time() 
    
    # Define the center of the bounding box
    center_lat = (lat_min + lat_max) / 2
    center_lon = (lon_min + lon_max) / 2

    # Construct the geospatial query
    latitude_query = DoublePoint.newRangeQuery("latitude", lat_min, lat_max)
    longitude_query = DoublePoint.newRangeQuery("longitude", lon_min, lon_max)

    boolean_query = BooleanQuery.Builder()
    boolean_query.add(latitude_query, BooleanClause.Occur.MUST)
    boolean_query.add(longitude_query, BooleanClause.Occur.MUST)
    query = boolean_query.build()

    # Set max_results heuristically based on N, assuming higher density in popular areas
    max_results = max(1000, 5 * N)  # Adjust based on expected density
    hits = searcher.search(query, max_results)

    results = []

    for hit in hits.scoreDocs:
        score = hit.score
        doc = searcher.doc(hit.doc)
        business_id = doc.get("business_id")
        name = doc.get("name") or "N/A"
        address = doc.get("address") or "N/A"
        city = doc.get("city") or "N/A"
        state = doc.get("state") or "N/A"
        postal_code = doc.get("postal_code") or "N/A"
        latitude = float(doc.get("latitude"))
        longitude = float(doc.get("longitude"))
        stars = float(doc.get("stars") or "0")
        review_count = int(doc.get("review_count") or "0")

        # Calculate distance to the center of the bounding box
        distance_to_center = calculate_distance(center_lat, center_lon, latitude, longitude)

        # Custom scoring formula
        if distance_to_center == 0:
            distance_to_center = 0.001  # Avoid division by zero

        adjusted_score = (10 / distance_to_center) * (1 + stars / 5.0) * (1 + review_count / 100.0) + score 

        result = {
            "business_id": business_id,
            "name": name,
            "address": address,
            "city": city,
            "state": state,
            "postal_code": postal_code,
            "stars": stars,
            "review_count": review_count,
            "distance_to_center": distance_to_center,
            "latitude": latitude,
            "longitude": longitude,
            "lucene_score": score,
            "adjusted_score": adjusted_score
        }
        results.append(result)

    # Sort results by the adjusted score in descending order
    results.sort(key=lambda x: x['adjusted_score'], reverse=True)
    
    end_time = time.time()
    print(f"\nQuery took {end_time - start_time:.4f} seconds")

    # Return the top N results
    return results[:N]

def get_reviews_by_user(searcher, user_id):
    """
    Retrieve all reviews for a given user ID.
    """
    query = TermQuery(Term("user_id", user_id))
    hits = searcher.search(query, 10000)  # Adjust the limit if needed
    reviews = []
    for hit in hits.scoreDocs:
        doc = searcher.doc(hit.doc)
        business_id =  doc.get("business_id")
        # Use TermQuery to exactly match the business_id field
        business_query = TermQuery(Term("business_id", business_id))
        business_hits = searcher.search(business_query, 1)

        if business_hits.totalHits.value > 0:
            business_doc = searcher.doc(business_hits.scoreDocs[0].doc)
            business_name = business_doc.get("name") or "N/A"
            latitude = float(business_doc.get("latitude") or 0)
            longitude = float(business_doc.get("longitude") or 0)

        reviews.append({
            "business_id": doc.get("business_id"),
            "business_name": business_name or 'N/A',
            "review_text": doc.get("review_text"),
            "latitude": float(latitude or 0),
            "longitude": float(longitude or 0),
        })

    return reviews

def plot_user_review_distribution(searcher):
    """
    Plot distribution of review counts per user.
    """
    reader = searcher.getIndexReader()
    review_counts = Counter()
    visited_review_ids = set()  # Track visited review IDs to avoid duplicates

    # Iterate over all documents and count reviews per user, avoiding duplicates
    print(reader.maxDoc())
    for i in range(reader.maxDoc()):
        doc = reader.document(i)
        review_id = doc.get("review_id")
        
        # Skip this document if the review ID has already been processed
        if review_id in visited_review_ids:
            continue
        
        # Mark this review ID as visited
        visited_review_ids.add(review_id)
        
        # Increment the count for this user ID
        user_id = doc.get("user_id")
        if user_id:
            review_counts[user_id] += 1

    # Prepare data for plotting
    review_frequency = Counter(review_counts.values())
    print(review_frequency)

    x = list(review_frequency.keys())
    y = list(review_frequency.values())

    y_log = [math.log(y_val) for y_val in y]

    plt.figure(figsize=(10, 5))
    plt.bar(x, y_log, label="Log Smoothed")  # Scatter plot to avoid line connections
    plt.xlabel('Number of Reviews')
    plt.ylabel('Number of Users (Log Smoothed)')
    plt.title('User Review Contribution Distribution')
    plt.grid(True)
    plt.show()

def calculate_bounding_box(reviews):
    """
    Calculate the bounding box for the geographical locations of businesses reviewed by a user.
    """
    latitudes = [r["latitude"] for r in reviews if r["latitude"] != 0]
    longitudes = [r["longitude"] for r in reviews if r["longitude"] != 0]

    if latitudes and longitudes:
        return {
            "lat_min": min(latitudes),
            "lat_max": max(latitudes),
            "lon_min": min(longitudes),
            "lon_max": max(longitudes),
        }
    return None

def get_top_words_and_phrases(reviews, top_n=10):
    """
    Extract the top N most frequent words and phrases from a user's reviews.
    """
    stopwords_list = list(stop_words.STOP_WORDS)

    stopwords_java_list = Arrays.asList(stopwords_list)
    stopwords_set = CharArraySet(stopwords_java_list, True) 

    analyzer = StandardAnalyzer(stopwords_set)
    token_count = Counter()
    phrase_count = Counter()

    for review in reviews:
        text = review["review_text"]
        token_stream = analyzer.tokenStream("review_text", text)
        token_stream.reset()

        terms = []
        while token_stream.incrementToken():
            term = token_stream.getAttribute(CharTermAttribute.class_).toString()
            terms.append(term)
            token_count[term] += 1

        token_stream.end()
        token_stream.close()

        # Count two-word phrases for simplicity
        phrases = [" ".join(terms[i:i+2]) for i in range(len(terms)-1)]
        phrase_count.update(phrases)

    return {
        "top_words": token_count.most_common(top_n),
        "top_phrases": phrase_count.most_common(top_n),
    }

def get_representative_sentences(reviews, top_n=3):
    """
    Extract representative sentences from the user's reviews.
    """
    all_sentences = []
    for review in reviews:
        sentences = review["review_text"].split(".")
        all_sentences.extend(sentences)

    index_user_documents("index_temp_user", all_sentences)

    tf_idf_vectors = calculate_tfidf("index_temp_user")

    sentence_num_sim = {}
    for sentence in tf_idf_vectors.keys():
        count = 0
        target_vector = tf_idf_vectors[sentence]

        for vector in tf_idf_vectors.values():
            similarity = cosine_similarity(target_vector, vector)
            if similarity >= 0.9:
                count += 1
        
        sentence_num_sim[sentence] = count

    sorted_sentences = dict(sorted(sentence_num_sim.items(), key=lambda item: item[1]))
    sorted_sentences.pop('')

    # Filter out sentences that contain only numbers
    top_sentences = [s for s in sorted_sentences.keys() if not re.fullmatch(r'\d+', s.strip())]
    top_3_sentences = top_sentences[:3]
    
    # Add a message if fewer than `top_n` sentences are available
    if len(top_3_sentences) < 3:
        top_3_sentences.append("No more sentences for this review")

    shutil.rmtree("index_temp_user", ignore_errors=True)
    return top_3_sentences

def index_user_documents(index_dir, documents):

    # Create the index directory
    if not os.path.exists(index_dir):
        os.makedirs(index_dir)

    # Set up index writer
    index = FSDirectory.open(Paths.get(index_dir))
    # index = FSDirectory.open(File(index_dir))
    analyzer = StandardAnalyzer()
    config = IndexWriterConfig(analyzer)
    writer = IndexWriter(index, config)

    # Add documents to the index
    for doc_id, content in enumerate(documents):
        doc = Document()
        doc.add(StringField("id", str(doc_id), StringField.Store.YES))
        doc.add(StringField("content", content, StringField.Store.YES))
        writer.addDocument(doc)

    writer.close()

def calculate_tfidf(index_dir):
    index = FSDirectory.open(Paths.get(index_dir))
    reader = DirectoryReader.open(index)

    # Total number of documents
    total_docs = reader.numDocs()
    doc_tfidf = {}

    for doc_id in range(total_docs):
        doc = reader.document(doc_id)
        terms = doc.getField("content").stringValue().split()  # Split into terms

        tf = {}
        for term in terms:
            tf[term] = tf.get(term, 0) + 1  # Term Frequency

        # Calculate TF-IDF
        tfidf_vector = {}
        for term, frequency in tf.items():
            # Calculate term frequency
            tf_value = frequency / len(terms)

            # Calculate document frequency
            term_docs = reader.docFreq(Term("content", term))
            idf_value = log(total_docs / (1 + term_docs))  # Using log smoothing

            # TF-IDF calculation
            tfidf_vector[term] = tf_value * idf_value

        
        doc_tfidf[doc.getField("content").stringValue()] = tfidf_vector

    return doc_tfidf

def cosine_similarity(vec1_dict, vec2_dict):
    vec1 = list(vec1_dict.values())
    vec2 = list(vec2_dict.values())
    
    dot = sum(vec1 * vec2 for vec1, vec2 in zip(vec1, vec2))
    mag1 = math.sqrt(sum(a ** 2 for a in vec1))
    mag2 = math.sqrt(sum(a ** 2 for a in vec2))
    
    if mag1 == 0 or mag2 == 0:  # Handle zero magnitude vectors
        return 0.0
    
    return dot / (mag1 * mag2)

def generate_user_review_summary(searcher, user_id):
    """
    Generate a summary of reviews for a specific user.
    """
    reviews = get_reviews_by_user(searcher, user_id)
    if not reviews:
        print(f"No reviews found for user ID: {user_id}")
        return

    # Number of reviews
    num_reviews = len(reviews)
    print(f"User {user_id} has contributed {num_reviews} reviews.")

    # Bounding box
    bounding_box = calculate_bounding_box(reviews)
    if bounding_box:
        print("User's review activity bounding box:")
        print(f"Latitude min: {bounding_box['lat_min']}, max: {bounding_box['lat_max']}")
        print(f"Longitude min: {bounding_box['lon_min']}, max: {bounding_box['lon_max']}")
    else:
        print("No geolocation data available for bounding box calculation.")

    # Top words and phrases
    top_words_phrases = get_top_words_and_phrases(reviews)
    print("Top 10 words:")
    for word, count in top_words_phrases["top_words"]:
        print(f"{word}: {count}")

    print("\nTop 10 phrases:")
    for phrase, count in top_words_phrases["top_phrases"]:
        print(f"{phrase}: {count}")

    # Representative sentences
    representative_sentences = get_representative_sentences(reviews)
    print("\nRepresentative sentences:")
    for sentence in representative_sentences:
        print(f"- {sentence.strip()}")

def print_search_results(hits, searcher, search_type):
    """
    Prints search results in a more informative manner, including star distribution and other details.
    """
    if not hits:
        print("No results found.")
        return
   
    print("\nResults:")
    
    for result in hits:
        print(f"Name: {result['name']}")
        print(f"Business ID: {result['business_id']}")
        
        if search_type == "business" or search_type == "geospatial":
            print(f"Stars: {result['stars']}")
            print(f"Review Count: {result['review_count']}")
            print(f"Address: {result['address']}, City: {result['city']}, State: {result['state']}")
            print(f"Postal Code: {result['postal_code']}")
        
        if search_type == "review":
            print(f"Stars: {result['stars']}")
            print(f"User ID: {result['user_id']}")
            print(f"Review ID: {result['review_id']}")
            print(f"Review Text: {result['review_text']}")
            print(f"Useful: {result['useful']}, Funny: {result['funny']}, Cool: {result['cool']}")
        
        if search_type == "geospatial":
            print(f"Latitude: {result['latitude']}, Longitude: {result['longitude']}")
            print(f"Distance to Center: {result['distance_to_center']:.2f} km")
        
        # Print adjusted score for geospatial and review searches
        print(f"Lucene Score: {result.get('score', result['lucene_score']):.10f}")
        print(f"Adjusted Score: {result.get('score', result['adjusted_score']):.10f}")
        print("-" * 40)

def get_business(bid, searcher):
    reader = searcher.getIndexReader()

    bool_query = BooleanQuery.Builder()
    doctype_business = TermQuery(Term('doc_type', 'business'))
    doctype_cluster = TermQuery(Term('doc_type', 'cluster'))
    query_bid = TermQuery(Term('business_id', bid))
    bool_query.add(doctype_business, BooleanClause.Occur.SHOULD)
    bool_query.add(doctype_cluster, BooleanClause.Occur.SHOULD)
    bool_query.add(query_bid, BooleanClause.Occur.MUST)
    query = bool_query.build()
    hits = searcher.search(query, 2)

    pri, cluster, cluster_size = None, None, None
    for hit in hits.scoreDocs:
        doc = reader.document(hit.doc)
        if doc.get('doc_type') == 'business':
            pri = doc
        else:
            cluster = doc.get('cluster')
            term = Term('cluster', cluster)
            cluster_size = reader.docFreq(term)
    return pri, cluster, cluster_size

def get_cluster(cluster, searcher, top_n):
    reader = searcher.getIndexReader()
    bool_query = BooleanQuery.Builder()
    cluster_term = Term('cluster', cluster)
    doctype_qry = TermQuery(Term('doc_type', 'cluster'))
    cluster_qry = TermQuery(cluster_term)
    bool_query.add(doctype_qry, BooleanClause.Occur.MUST)
    bool_query.add(cluster_qry, BooleanClause.Occur.MUST)
    query = bool_query.build()
    n = reader.docFreq(cluster_term)
    hits = searcher.search(query,n)
    docs = []
    for hit in hits.scoreDocs:
        doc = reader.document(hit.doc)
        bid = doc.get('business_id')
        doc = get_business(bid, searcher)[0]
        docs.append(doc)
    docs = np.array(docs)
    return docs


def random_business_cluster_reco(searcher, n_biz):
    '''
    Randomly select a cluster
    '''
    reader = searcher.getIndexReader()
    n = reader.maxDoc()
    query = TermQuery(Term('doc_type', 'cluster'))
    hits = searcher.search(query, n)
    clusters = set()
    for hit in hits.scoreDocs:
        doc = reader.document(hit.doc)
        cluster = doc.get('cluster')
        clusters.add(cluster)
    clusters = list(clusters)
    n = len(clusters)
    cl = str(int(np.random.choice(n)))
    reco = get_cluster(cl, searcher, n_biz)
    return reco, cl

def common_cluster_reco(hits,n_sim, searcher):
    '''
    Filter businesses by the cluster of the businesses in hits, while omitted those seen businesses
    '''
    reader = searcher.getIndexReader()
    n_hits = len(hits)
    n_candidates = 0
    bool_query = BooleanQuery.Builder()
    bool_query.add(TermQuery(Term('doc_type', 'cluster')), BooleanClause.Occur.MUST)

    original_docs = []

    for hit in hits:
        bid = hit['business_id']
        doc, cluster, cluster_size = get_business(bid, searcher)
        original_docs.append(doc)
        n_candidates += cluster_size
        bool_query.add(TermQuery(Term('cluster', cluster)), BooleanClause.Occur.SHOULD)
        bool_query.add(TermQuery(Term('business_id', bid)), BooleanClause.Occur.MUST_NOT)

    query = bool_query.build() 
    hits = searcher.search(query, n_candidates)

    candidates = []

    for hit in hits.scoreDocs:
        doc = reader.document(hit.doc)
        candidates.append(doc.get('business_id'))
   
    candidates = np.array([get_business(bid, searcher)[0] for bid in candidates])
    orignal_docs = np.array(original_docs)
    return candidates, original_docs


def business_reco(hits, n_sim, searcher):
    '''
    Recommend businesses.
    If there have been previously viewed businesses (business by name task) , pick from those clusters
    Otherwise, randomly pick a cluster
    '''
    candidates = None
    original_docs = None
    if not hits:
        reco, cl = random_business_cluster_reco(searcher, n_sim)
        candidates = reco
    else:
        reco, original_docs = common_cluster_reco(hits,n_sim, searcher)
        candidates = reco
     
    rating = np.array([float(doc.get('stars')) for doc in candidates])
    top_n = np.argsort(-rating)[:n_sim]
    reco = candidates[top_n].tolist() 

    return reco, original_docs

def print_recommendations(reco_out, n):
    '''
    Format output for application task
    '''
    reco, orig = reco_out
    
    if orig:
        print("-" * 40)
        print('Because you looked at:')
        print("-" * 40)
        for doc in orig:
            print(doc.get('name'))
            print(doc.get('categories'))
            print(doc.get('stars'))
            print(doc.get('address'))
            print("-" * 40)
    else:
        print("-" * 40)
    print('Maybe consider paying a visit to:')
    print("-" * 40)
    for i, doc in enumerate(reco):
        print(f'{i} / {n}')
        print(doc.get('name'))
        print(doc.get('categories'))
        print(doc.get('stars'))
        print(doc.get('address'))
        print("-" * 40)

    print('End of Results')
    print("-" * 40)


def prompt_for_N():
    while True:
        try:
            N = int(input("Enter the number of results you want (N): "))
            if N <= 0:
                print("Number of results must be a positive integer. Please try again.")
                continue  # Prompt again for a positive integer
            return N # Exit the loop if N is valid
        except ValueError:
            print("Invalid input. Please enter a positive integer for the number of results.")
            # continue is not necessary here; it will loop back automatically

def terminal_ui(searcher):
    """
    Terminal UI for selecting the type of search and interacting with the search engine.
    """
    choices = [
        "1. Search by Business Name",
        "2. Search by Review Text",
        "3. Geospatial Search (Bounding Box)",
        "4. User Review Summary",
        "5. Distribution of reviews contributed by the users",
        "6. Recommend Similar Businesses",
        "7. Exit"
    ]
   
    last_business_hits = None

    while True:
        print("\nSearch Options:")
        num_choices = len(choices) 
        for choice in choices:
            print(choice)
        choice = input(f"Enter the type of search you want (1-{num_choices}): ")
    
        if not choice.isdigit() or not (1 <= int(choice) <= num_choices):
            print(f"Invalid choice. Please enter a number between 1 and {num_choices}.")
            continue

        choice = int(choice)

        if choice == num_choices:
            print("Exiting...")
            break
        
        elif choice == 6:
            N = prompt_for_N()
            hits = last_business_hits
            reco_out = business_reco(hits, N, searcher)
            print_recommendations(reco_out, N)
            
        elif choice == 4:
            user_id = input("Enter the user ID: ").strip()
            generate_user_review_summary(searcher, user_id)

        elif choice == 5:
            plot_user_review_distribution(searcher)

        elif choice == 1:
            N = prompt_for_N()
            business_name = input("Enter the business name: ").strip()
            if len(business_name) == 0:
                print("Business name cannot be empty. Please enter a valid business name.")
                continue
            elif len(business_name) < 3:
                print("Business name is too short. Please enter at least 3 characters.")
                continue
            hits = search_by_business_name(searcher, business_name, N)
            last_business_hits = hits
            print_search_results(hits, searcher, "business") if hits else print("No results found for this business name.")
            
            
        elif choice == 2:
            N = prompt_for_N()
            review_text = input("Enter the review text: ").strip()
            if len(review_text) == 0:
                print("Review text cannot be empty. Please enter valid text.")
                continue
            elif len(review_text) < 3:
                print("Review text is too short. Please enter at least 3 characters.")
                continue
            hits = search_by_review_text_with_business(searcher, review_text, N)
            print_search_results(hits, searcher, "review") if hits else print("No reviews found matching this text.")

        elif choice == 3:
            N = prompt_for_N()
            try:
                # Input and validate latitude and longitude ranges
                lat_min = float(input("Enter minimum latitude: "))
                lat_max = float(input("Enter maximum latitude: "))
                lon_min = float(input("Enter minimum longitude: "))
                lon_max = float(input("Enter maximum longitude: "))
                
                # Validate latitude and longitude bounds
                if not (-90 <= lat_min <= 90) or not (-90 <= lat_max <= 90):
                    print("Latitude values must be between -90 and 90.")
                    continue
                if not (-180 <= lon_min <= 180) or not (-180 <= lon_max <= 180):
                    print("Longitude values must be between -180 and 180.")
                    continue
                if lat_min > lat_max:
                    print("Minimum latitude cannot be greater than maximum latitude.")
                    continue
                if lon_min > lon_max:
                    print("Minimum longitude cannot be greater than maximum longitude.")
                    continue

                # Perform geospatial search
                hits = geospatial_search(searcher, lat_min, lat_max, lon_min, lon_max, N)
                print_search_results(hits, searcher, "geospatial") if hits else print("No businesses found within this bounding box.")
            except ValueError:
                print("Invalid input for latitude or longitude. Please try again.")
            except Exception as e:
                print(f"Error during search: {e}")


if __name__ == "__main__":
    # Initialize the Lucene JVM
    lucene.initVM(vmargs=['-Djava.awt.headless=true'])

    # Define the path to the index directory
    index_directory = "./index"
    secondary_index_directory = "./secondary_index"
    # Open the index directory
    reader = DirectoryReader.open(FSDirectory.open(Paths.get(index_directory)))
    secondary_reader = DirectoryReader.open(FSDirectory.open(Paths.get(secondary_index_directory)))
    reader = MultiReader([reader, secondary_reader])
    searcher = IndexSearcher(reader)

    # Start the terminal UI for searching
    terminal_ui(searcher)
