from unittest.mock import Mock, call, patch

from haystack import Document

from llmrag.dense_experiment import run_dense_experiment


def test_run_dense_experiment_connects_components(tmp_path):
    chunks = [Document(content="Paper text")]
    cases = [{"id": "q1", "question": "A question?", "answerable": True}]
    document_vectors = [[1.0, 0.0]]
    query_vectors = [[0.0, 1.0]]
    expected_results = [{"id": "q1", "retrieved": []}]

    client = Mock()
    document_cache = tmp_path / "documents.json"
    query_cache = tmp_path / "queries.json"

    with (
        patch(
            "llmrag.dense_experiment.load_or_embed_texts",
            autospec=True,
        ) as embed,
        patch(
            "llmrag.dense_experiment.build_dense_retriever",
            autospec=True,
        ) as build,
        patch(
            "llmrag.dense_experiment.collect_dense_retrieval_results",
            autospec=True,
        ) as collect,
    ):
        embed.side_effect = [document_vectors, query_vectors]
        collect.return_value = expected_results

        results = run_dense_experiment(
            chunks,
            cases,
            client=client,
            document_cache=document_cache,
            query_cache=query_cache,
            top_k=2,
        )

        assert embed.call_args_list == [
            call(
                texts=["Paper text"],
                cache_path=document_cache,
                client=client,
                task_type="RETRIEVAL_DOCUMENT",
            ),
            call(
                texts=["A question?"],
                cache_path=query_cache,
                client=client,
                task_type="RETRIEVAL_QUERY",
            ),
        ]

        build.assert_called_once_with(chunks, document_vectors)

        collect.assert_called_once_with(
            cases,
            query_vectors,
            retriever=build.return_value,
            top_k=2,
        )

        assert results == expected_results
