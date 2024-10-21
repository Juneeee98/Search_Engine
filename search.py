import lucene
import time
import math
from org.apache.lucene.store import FSDirectory
from org.apache.lucene.index import DirectoryReader, Term
from org.apache.lucene.analysis.standard import StandardAnalyzer
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

def search_by_business_name(searcher, analyzer, query_string, N):
    """
    Searches for businesses by their name and provides star distribution for each result.
    Custom scoring is applied based on star ratings and review count.
    """
    start_time = time.time()

    # Execute keyword search for business name
    query = QueryParser("name", analyzer).parse(query_string)

    # Retrieve ALL matching documents first (set a high max limit)
    max_results = max(50, 2 * N)  # Set this high to ensure all matching docs are retrieved
    hits = searcher.search(query, max_results)
    end_time = time.time()

    print(f"\nQuery took {end_time - start_time:.4f} seconds")

    results = []
    seen_business_ids = set()  # To track unique business IDs

    for hit in hits.scoreDocs:
        doc = searcher.doc(hit.doc)
        business_id = doc.get("business_id")
        
        # Skip if this business_id has already been seen
        if business_id in seen_business_ids:
            continue

        name = doc.get("name") or "N/A"
        stars = float(doc.get("stars") or "0")  # Convert stars to float
        review_count = int(doc.get("review_count") or "0")  # Convert review count to int
        score = hit.score

        # Custom scoring logic: combine keyword relevance (default Lucene score) with stars and review count
        if review_count == 0:  # Avoid dividing by zero
            review_count = 1

        # Adjust score based on stars and review count, with scaling factors
        adjusted_score = score * (1 + (stars / 5.0)) * (1 + (review_count / 100.0))

        result = {
            "business_id": business_id,
            "name": name,
            "stars": stars,
            "review_count": review_count,
            "score": adjusted_score  # Use custom adjusted score
        }
        results.append(result)

        # Mark this business_id as seen to avoid duplicates
        seen_business_ids.add(business_id)

    # Sort results by the adjusted score in descending order
    results.sort(key=lambda x: x['score'], reverse=True)

    # Return only the top N results after filtering
    return results[:N]

def search_by_review_text_with_business(searcher, analyzer, query_string, N):
    """
    Searches for reviews by keywords in the review text and retrieves associated business names.
    Custom scoring is applied based on keyword relevance, usefulness, coolness, funniness, and business ratings.
    """
    start_time = time.time()

    # Retrieve all matching documents (using a large value for maxResults)
    query = QueryParser("review_text", analyzer).parse(query_string)
    max_results = max(1000, 2 * N)  # Retrieve all potential matches
    hits = searcher.search(query, max_results)
    end_time = time.time()

    print(f"\nQuery took {end_time - start_time:.4f} seconds")

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

        # Custom scoring:
        # Adjust the score based on usefulness, coolness, funniness, and business star rating
        adjusted_score = score * (
            1 + (useful * useful_weight / 10.0) + (cool * cool_weight / 10.0) + (funny * funny_weight / 10.0)
        ) * (1 + stars / 5.0)

        result = {
            "business_id": business_id,
            "review_id": review_id,  # Adding review_id to track unique results
            "name": business_name,
            "review_text": review_text,
            "useful": useful,
            "cool": cool,
            "funny": funny,
            "stars": stars,
            "score": adjusted_score  # Use custom adjusted score
        }

        # Separate reviews with valid business names
        if business_name != "N/A":
            results_with_business_name.append(result)
        else:
            results_without_business_name.append(result)

        # Add the review_id to the seen set to prevent duplicates
        seen_review_ids.add(review_id)

    # Combine prioritized results (with business name first)
    prioritized_results = results_with_business_name + results_without_business_name

    # Sort by the adjusted score in descending order
    prioritized_results.sort(key=lambda x: x['score'], reverse=True)

    # Return only the top N results after filtering
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

    # Retrieve all matching documents (use a large number for maxResults to allow for custom scoring)
    max_results = 1000  # Retrieve all matches within the bounding box
    hits = searcher.search(query, max_results)
    end_time = time.time()

    print(f"\nQuery took {end_time - start_time:.4f} seconds")

    results = []

    for hit in hits.scoreDocs:
        doc = searcher.doc(hit.doc)
        business_id = doc.get("business_id")
        name = doc.get("name") or "N/A"
        address = doc.get("address") or "N/A"
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

        adjusted_score = (1 / distance_to_center) * (1 + stars / 5.0) * (1 + review_count / 100.0)

        result = {
            "business_id": business_id,
            "name": name,
            "address": address,
            "postal_code": postal_code,
            "stars": stars,
            "review_count": review_count,
            "distance_to_center": distance_to_center,
            "score": adjusted_score
        }
        results.append(result)

    # Sort results by the adjusted score in descending order
    results.sort(key=lambda x: x['score'], reverse=True)

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
        print(f"Business ID: {result['business_id']}")
        print(f"Name: {result['name']}")
        
        if search_type == "business" or search_type == "geospatial":
            print(f"Stars: {result['stars']}")
            print(f"Review Count: {result['review_count']}")
        
        if search_type == "review":
            print(f"Review Text: {result['review_text']}")
            print(f"Useful: {result['useful']}, Funny: {result['funny']}, Cool: {result['cool']}")
        
        if search_type == "geospatial":
            print(f"Address: {result['address']}, Postal Code: {result['postal_code']}")
            print(f"Distance to Center: {result['distance_to_center']:.2f} km")
        
        # Print adjusted score for geospatial and review searches
        print(f"Adjusted Score: {result.get('score', result['score']):.10f}")
        print("-" * 40)

def terminal_ui(searcher, analyzer):
    """Terminal UI for selecting the type of search and interacting with the search engine."""
    while True:
        print("\nSearch Options:")
        print("1. Search by Business Name")
        print("2. Search by Review Text")
        print("3. Geospatial Search (Bounding Box)")
        print("4. Exit")
        choice = input("Enter the type of search you want (1-4): ").strip()
    
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
            hits = search_by_business_name(searcher, analyzer, business_name, N)
            print_search_results(hits, searcher, "business") if hits else print("No results found for this business name.")
            
            
        elif choice == 2:
            review_text = input("Enter the review text: ").strip()
            if len(review_text) == 0:
                print("Review text cannot be empty. Please enter valid text.")
                continue
            elif len(review_text) < 3:
                print("Review text is too short. Please enter at least 3 characters.")
                continue
            hits = search_by_review_text_with_business(searcher, analyzer, review_text, N)
            print_search_results(hits, searcher, "review") if hits else print("No reviews found matching this text.")

        elif choice == 3:
            try:
                lat_min = float(input("Enter minimum latitude: "))
                lat_max = float(input("Enter maximum latitude: "))
                lon_min = float(input("Enter minimum longitude: "))
                lon_max = float(input("Enter maximum longitude: "))
                if lat_min > lat_max:
                    print("Minimum latitude cannot be greater than maximum latitude.")
                    continue
                if lon_min > lon_max:
                    print("Minimum longitude cannot be greater than maximum longitude.")
                    continue
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
    analyzer = StandardAnalyzer()
    searcher = IndexSearcher(DirectoryReader.open(FSDirectory.open(Paths.get(index_directory))))

    # Start the terminal UI for searching
    terminal_ui(searcher, analyzer)
