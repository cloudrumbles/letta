import os
import pytest
from letta.embeddings import VoyageEmbedding

@pytest.mark.skipif(
    not os.getenv("VOYAGE_API_KEY"),
    reason="VOYAGE_API_KEY environment variable is not set",
)
def test_get_text_embedding():
    api_key = os.getenv("VOYAGE_API_KEY")
    voyage_embedding = VoyageEmbedding(api_key=api_key)
    text = "Sample text"
    embedding = voyage_embedding.get_text_embedding(text)
    assert isinstance(embedding, list)
    assert all(isinstance(x, float) for x in embedding)
    assert len(embedding) > 0
