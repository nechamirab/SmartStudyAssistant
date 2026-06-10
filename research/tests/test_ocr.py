import unittest

from ingestion.ocr_loader import OCRDependencyError, OCRLoader


class OCRTests(unittest.TestCase):
    def test_missing_file_is_clear(self):
        with self.assertRaises(FileNotFoundError):
            OCRLoader().extract("does-not-exist.pdf")

    def test_dependency_error_message_is_actionable(self):
        message = str(OCRDependencyError("install pytesseract"))
        self.assertIn("pytesseract", message)


if __name__ == "__main__":
    unittest.main()
