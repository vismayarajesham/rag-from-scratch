from fpdf import FPDF

pdf = FPDF()
pdf.add_page()
pdf.set_font("Helvetica", size=12)

text = """Nova Robotics - NovaBot X2 Service Manual

Release: The NovaBot X2 was released in September 2025. It is the successor
to the NovaBot X1 and weighs 4.2 kg.

Water resistance: The NovaBot X2 has an IP54 rating. It can survive splashes
of water and rain, but it must not be submerged in water. Do not use it in
swimming pools or bathrooms with running showers.

Battery and charging: The NovaBot X2 uses a 6000 mAh battery. On a full
charge it runs for approximately 8 hours of normal use. A full charge takes
120 minutes using the 30V fast charger included in the box.

Warranty: The NovaBot X2 comes with a 24 month standard warranty. This can
be extended up to 48 months if the extension is purchased within 60 days of
the original purchase date.

Price: The NovaBot X2 is priced at 49,999 rupees in India. Educational
institutions receive a 15 percent discount on bulk orders of 10 units or more.

New features: Voice control now supports 9 Indian languages. The X2 also
introduces a self-emptying dustbin and automatic carpet-boost suction."""

pdf.multi_cell(0, 7, text)
pdf.output("data/novabot_x2_manual.pdf")
print("Created data/novabot_x2_manual.pdf")