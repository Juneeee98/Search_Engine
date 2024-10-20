import lucene
import time
from collections import defaultdict
from org.apache.lucene.store import FSDirectory
from org.apache.lucene.index import DirectoryReader
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.search import IndexSearcher, BooleanQuery, BooleanClause
from org.apache.lucene.document import DoublePoint
from java.nio.file import Paths

# Initialize the Lucene JVM
lucene.initVM(vmargs=['-Djava.awt.headless=true'])

def search_by_business_name(searcher, analyzer, query_string, N):
    """Searches for businesses by their name and provides star distribution for each result."""
    start_time = time.time()
    query = QueryParser("name_index", analyzer).parse(query_string)
    hits = searcher.search(query, N)
    end_time = time.time()

    print(f"\nQuery took {end_time - start_time:.4f} seconds")

    results = []
    for hit in hits.scoreDocs:
        doc = searcher.doc(hit.doc)
        business_id = doc.get("business_id")
        name = doc.get("name") or "N/A"
        stars = doc.get("stars") or "N/A"
        score = hit.score

        # Collect star distribution
        star_distribution = defaultdict(int)
        if stars != "N/A":
            star_distribution[int(float(stars))] += 1

        result = {
            "business_id": business_id,
            "name": name,
            "star_distribution": star_distribution,
            "score": score
        }
        results.append(result)

    return results

def search_by_review_text_with_business(searcher, analyzer, query_string, N):
    """Searches for reviews by keywords in the review text and retrieves associated business names."""
    start_time = time.time()
    query = QueryParser("review_text", analyzer).parse(query_string)
    hits = searcher.search(query, N)
    end_time = time.time()

    print(f"\nQuery took {end_time - start_time:.4f} seconds")
    results_with_business_name = []
    results_without_business_name = []

    for hit in hits.scoreDocs:
        doc = searcher.doc(hit.doc)
        business_id = doc.get("business_id")
        review_text = doc.get("review_text") or "N/A"
        useful = doc.get("useful") or "0"
        funny = doc.get("funny") or "0"
        cool = doc.get("cool") or "0"
        score = hit.score

        # Fetch the business document by business_id
        business_query = QueryParser("business_id", analyzer).parse(business_id)
        business_hits = searcher.search(business_query, 1)
        business_name = "N/A"
        star_distribution = defaultdict(int)
        if business_hits.totalHits.value > 0:
            business_doc = searcher.doc(business_hits.scoreDocs[0].doc)
            business_name = business_doc.get("name") or "N/A"
            stars = business_doc.get("stars") or "N/A"
            if stars != "N/A":
                star_distribution[int(float(stars))] += 1

        result = {
            "business_id": business_id,
            "name": business_name,
            "review_text": review_text,
            "useful": useful,
            "funny": funny,
            "cool": cool,
            "star_distribution": star_distribution,
            "score": score
        }

        # Separate reviews with valid business names
        if business_name != "N/A":
            results_with_business_name.append(result)
        else:
            results_without_business_name.append(result)

    # Combine prioritized results (with business name first)
    prioritized_results = results_with_business_name + results_without_business_name
    return prioritized_results[:N]

def geospatial_search(searcher, lat_min, lat_max, lon_min, lon_max, N):
    """Searches for businesses within a given geospatial bounding box."""
    start_time = time.time()
    latitude_query = DoublePoint.newRangeQuery("latitude", lat_min, lat_max)
    longitude_query = DoublePoint.newRangeQuery("longitude", lon_min, lon_max)

    boolean_query = BooleanQuery.Builder()
    boolean_query.add(latitude_query, BooleanClause.Occur.MUST)
    boolean_query.add(longitude_query, BooleanClause.Occur.MUST)
    query = boolean_query.build()

    hits = searcher.search(query, N)
    end_time = time.time()

    print(f"\nQuery took {end_time - start_time:.4f} seconds")

    results = []
    for hit in hits.scoreDocs:
        doc = searcher.doc(hit.doc)
        business_id = doc.get("business_id")
        name = doc.get("name") or "N/A"
        address = doc.get("address") or "N/A"
        postal_code = doc.get("postal_code") or "N/A"
        stars = doc.get("stars") or "N/A"
        score = hit.score

        # Collect star distribution
        star_distribution = defaultdict(int)
        if stars != "N/A":
            star_distribution[int(float(stars))] += 1

        result = {
            "business_id": business_id,
            "name": name,
            "address": address,
            "postal_code": postal_code,
            "star_distribution": star_distribution,
            "score": score
        }
        results.append(result)

    return results

def print_search_results(hits, search_type):
    """Prints search results in a more informative manner, including star distribution."""
    if not hits:
        print("No results found.")
        return
    
    print("\nResults:")
    
    for result in hits:
        print(f"Business ID: {result['business_id']}")
        print(f"Name: {result['name']}")
        
        if search_type == "business" or search_type == "geospatial":
            print("Star Distribution:")
            for stars, count in sorted(result["star_distribution"].items()):
                print(f"{stars} star{'s' if stars > 1 else ''}: {count}")
        
        if search_type == "review":
            print(f"Review Text: {result['review_text']}")
            print(f"Useful: {result['useful']}, Funny: {result['funny']}, Cool: {result['cool']}")
        
        if search_type == "geospatial":
            print(f"Address: {result['address']}, Postal Code: {result['postal_code']}")
        
        print(f"Score: {result['score']:.4f}")
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
            print_search_results(hits, "business") if hits else print("No results found for this business name.")
            
            
        elif choice == 2:
            review_text = input("Enter the review text: ").strip()
            if len(review_text) == 0:
                print("Review text cannot be empty. Please enter valid text.")
                continue
            elif len(review_text) < 3:
                print("Review text is too short. Please enter at least 3 characters.")
                continue
            hits = search_by_review_text_with_business(searcher, analyzer, review_text, N)
            print_search_results(hits, "review") if hits else print("No reviews found matching this text.")

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
                print_search_results(hits, "geospatial") if hits else print("No businesses found within this bounding box.")
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
