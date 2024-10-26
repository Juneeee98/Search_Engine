import lucene
import time
import math
from math import log10
from org.apache.lucene.analysis.tokenattributes import CharTermAttribute
from org.apache.lucene.store import FSDirectory
from org.apache.lucene.index import DirectoryReader, Term
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.analysis.en import EnglishAnalyzer
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.search import TermQuery, IndexSearcher, BooleanQuery, BooleanClause
from org.apache.lucene.document import DoublePoint
from java.nio.file import Paths

# Initialize the Lucene JVM
lucene.initVM(vmargs=['-Djava.awt.headless=true'])

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
        adjusted_score = score * log_star_weight * log_review_count_weight

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
        adjusted_score = score * (
            1 + log_useful_weight / 10.0 + log_cool_weight / 10.0 + log_funny_weight / 10.0
        ) * log_star_weight

        result = {
            "business_id": business_id,
            "review_id": review_id,  # Adding review_id to track unique results
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
            print(f"Review Text: {result['review_text']}")
            print(f"Useful: {result['useful']}, Funny: {result['funny']}, Cool: {result['cool']}")
        
        if search_type == "geospatial":
            print(f"Address: {result['address']}, City: {result['city']}, State: {result['state']}")
            print(f"Postal Code: {result['postal_code']}")
            print(f"Latitude: {result['latitude']}, Longitude: {result['longitude']}")
            print(f"Distance to Center: {result['distance_to_center']:.2f} km")
        
        # Print adjusted score for geospatial and review searches
        print(f"Lucene Score: {result.get('score', result['lucene_score']):.10f}")
        print(f"Adjusted Score: {result.get('score', result['adjusted_score']):.10f}")
        print("-" * 40)

def terminal_ui(searcher):
    """
    Terminal UI for selecting the type of search and interacting with the search engine.
    """
    while True:
        print("\nSearch Options:")
        print("1. Search by Business Name")
        print("2. Search by Review Text")
        print("3. Geospatial Search (Bounding Box)")
        print("4. Exit")
        choice = input("Enter the type of search you want (1-4): ")
    
        if not choice.isdigit() or not (1 <= int(choice) <= 4):
            print("Invalid choice. Please enter a number between 1 and 4.")
            continue

        choice = int(choice)

        if choice == 4:
            print("Exiting...")
            break

        while True:
            try:
                N = int(input("Enter the number of results you want (N): "))
                if N <= 0:
                    print("Number of results must be a positive integer. Please try again.")
                    continue  # Prompt again for a positive integer
                break  # Exit the loop if N is valid
            except ValueError:
                print("Invalid input. Please enter a positive integer for the number of results.")
                # continue is not necessary here; it will loop back automatically

        if choice == 1:
            business_name = input("Enter the business name: ").strip()
            if len(business_name) == 0:
                print("Business name cannot be empty. Please enter a valid business name.")
                continue
            elif len(business_name) < 3:
                print("Business name is too short. Please enter at least 3 characters.")
                continue
            hits = search_by_business_name(searcher, business_name, N)
            print_search_results(hits, searcher, "business") if hits else print("No results found for this business name.")
            
            
        elif choice == 2:
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
    # Define the path to the index directory
    index_directory = "./index"

    # Open the index directory
    searcher = IndexSearcher(DirectoryReader.open(FSDirectory.open(Paths.get(index_directory))))

    # Start the terminal UI for searching
    terminal_ui(searcher)