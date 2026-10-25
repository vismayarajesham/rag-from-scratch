"""STEP 12 check: what text did the PDF actually get?"""

from pypdf import PdfReader

text = PdfReader("data/novabot_x2_manual.pdf").pages[0].extract_text()

print(text)
print()
print("contains '24 month'? ", "24 month" in text)
print("contains '49,999'?  ", "49,999" in text)