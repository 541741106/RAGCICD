from llmrag.bm25_baseline import retrieve_examples


def test_segmentation_query_ranks_segmentation_first():
    documents = retrieve_examples("image segmentation", top_k=2)

    assert len(documents) == 2
    assert documents[0].meta["paper_id"] == "example-segmentation"


def test_bounding_boxes():
    documents = retrieve_examples("bounding boxes", top_k=1)

    assert len(documents) == 1
    assert documents[0].meta["paper_id"] == "example-detection"


def test_image_segmentation():
    documents = retrieve_examples("image segmentation", top_k=1)

    img_meta = {
        "paper_id": "example-segmentation",
        "paper_title": "Synthetic Segmentation Paper",
        "page_number": 1,
        "source": "synthetic",
    }

    assert len(documents) == 1
    assert documents[0].meta == img_meta
