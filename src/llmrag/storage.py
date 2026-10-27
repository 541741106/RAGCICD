from haystack.utils import Secret
from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore


def open_document_store() -> PgvectorDocumentStore:
    """Open the persistent store for the ResNet retrieval experiment."""
    return PgvectorDocumentStore(
        connection_string=Secret.from_env_var("PG_CONN_STR"),
        create_extension=False,
        table_name="resnet_gemini_768_v1",
        embedding_dimension=768,
        vector_function="cosine_similarity",
        recreate_table=False,
        search_strategy="exact_nearest_neighbor",
    )
