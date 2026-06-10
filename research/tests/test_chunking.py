import unittest

from core.models import DocumentPage
from services.chunk_service import ChunkService, ChunkingError


class ChunkingTests(unittest.TestCase):
    def test_chunking_preserves_page_metadata(self):
        chunks = ChunkService(20, 5).chunk_pages(
            [
                DocumentPage(
                    page_number=2,
                    text="alpha beta gamma delta epsilon zeta eta theta",
                    source_id="notes.pdf",
                    metadata={"course": "ai"},
                )
            ]
        )

        self.assertTrue(chunks)
        self.assertEqual(chunks[0].source_id, "notes.pdf")
        self.assertEqual(chunks[0].metadata["course"], "ai")

    def test_invalid_overlap_raises(self):
        with self.assertRaises(ChunkingError):
            ChunkService(100, 100)


if __name__ == "__main__":
    unittest.main()
