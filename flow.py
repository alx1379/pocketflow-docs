from pocketflow import Flow
# Import all node classes from nodes.py
from nodes import (
    FetchRepo,
    IdentifyAbstractions,  # Keep for backward compatibility
    IdentifyAbstractionsMap,  # Map phase
    IdentifyAbstractionsReduce,  # Reduce phase
    AnalyzeRelationships,
    OrderChapters,
    WriteChapters,
    CombineTutorial
)

def create_tutorial_flow():
    """Creates and returns the codebase tutorial generation flow."""

    # Instantiate nodes
    fetch_repo = FetchRepo()
    
    # Use Map-Reduce pattern for identifying abstractions (handles any number of files)
    identify_abstractions_map = IdentifyAbstractionsMap(max_retries=3, wait=10)
    identify_abstractions_reduce = IdentifyAbstractionsReduce(max_retries=5, wait=20)
    
    analyze_relationships = AnalyzeRelationships(max_retries=5, wait=20)
    order_chapters = OrderChapters(max_retries=5, wait=20)
    write_chapters = WriteChapters(max_retries=5, wait=20) # This is a BatchNode
    combine_tutorial = CombineTutorial()

    # Connect nodes in sequence
    fetch_repo >> identify_abstractions_map
    identify_abstractions_map >> identify_abstractions_reduce
    identify_abstractions_reduce >> analyze_relationships
    analyze_relationships >> order_chapters
    order_chapters >> write_chapters
    write_chapters >> combine_tutorial

    # Create the flow starting with FetchRepo
    tutorial_flow = Flow(start=fetch_repo)

    return tutorial_flow
