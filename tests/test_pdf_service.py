import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import fitz

from services.pdf_service import PdfExtractionError, PdfService


class PdfServiceTests(unittest.TestCase):
    def test_extracts_normal_text_pdf_without_ocr(self):
        with TemporaryDirectory() as tmpdir:
            pdf_path = Path(tmpdir) / "normal.pdf"
            self._write_text_pdf(pdf_path, "Normal PDF text for study notes.")

            pages = PdfService().extract_pages(pdf_path, ocr_mode="auto")

        self.assertEqual(len(pages), 1)
        self.assertIn("Normal PDF text", pages[0].text)
        self.assertEqual(pages[0].metadata["extraction_method"], "normal")
        self.assertFalse(pages[0].metadata["ocr_attempted"])

    def test_auto_ocrs_scanned_page_when_normal_text_is_empty(self):
        with TemporaryDirectory() as tmpdir:
            pdf_path = Path(tmpdir) / "scanned.pdf"
            self._write_image_only_pdf(pdf_path)

            with patch.object(PdfService, "_ocr_page", return_value="Scanned OCR text from page one."):
                pages = PdfService().extract_pages(pdf_path, ocr_mode="auto")

        self.assertEqual(len(pages), 1)
        self.assertEqual(pages[0].text, "Scanned OCR text from page one.")
        self.assertEqual(pages[0].page_number, 1)
        self.assertEqual(pages[0].metadata["extraction_method"], "ocr")
        self.assertTrue(pages[0].metadata["ocr_attempted"])

    def test_force_ocr_runs_even_when_normal_text_exists(self):
        with TemporaryDirectory() as tmpdir:
            pdf_path = Path(tmpdir) / "force.pdf"
            self._write_text_pdf(pdf_path, "Normal PDF text that should be replaced.")

            with patch.object(PdfService, "_ocr_page", return_value="Forced OCR text."):
                pages = PdfService().extract_pages(pdf_path, ocr_mode="force")

        self.assertEqual(pages[0].text, "Forced OCR text.")
        self.assertEqual(pages[0].metadata["extraction_method"], "ocr")
        self.assertTrue(pages[0].metadata["ocr_attempted"])

    def test_off_mode_reports_empty_scanned_pdf_without_ocr(self):
        with TemporaryDirectory() as tmpdir:
            pdf_path = Path(tmpdir) / "scanned.pdf"
            self._write_image_only_pdf(pdf_path)

            with self.assertRaises(PdfExtractionError) as context:
                PdfService().extract_pages(pdf_path, ocr_mode="off")

        self.assertIn("Try OCR mode auto or force", str(context.exception))

    @staticmethod
    def _write_text_pdf(pdf_path: Path, text: str) -> None:
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), text)
        doc.save(pdf_path)
        doc.close()

    @staticmethod
    def _write_image_only_pdf(pdf_path: Path) -> None:
        doc = fitz.open()
        page = doc.new_page()
        page.draw_rect(fitz.Rect(72, 72, 260, 160), color=(0, 0, 0), fill=(0.95, 0.95, 0.95))
        doc.save(pdf_path)
        doc.close()


if __name__ == "__main__":
    unittest.main()
